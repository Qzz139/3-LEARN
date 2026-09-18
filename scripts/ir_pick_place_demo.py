#!/usr/bin/env python3
"""One-shot EP infrared-triggered pick/place demo with independent YOLO logging."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import queue
import signal
import socket
import threading
import time


ROOT = Path(__file__).resolve().parent.parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def raw_to_sdk_degrees(raw):
    """DDS encoder: 1024 = 180 degrees; SDK action angle is centred at 180."""
    return int(round(float(raw) * 180.0 / 1024.0 - 180.0))


def sdk_degrees_to_raw(degrees):
    """Expected DDS encoder value, distinct from the action's wire encoding."""
    return int(round((int(degrees) + 180) * 1024.0 / 180.0))


def sdk_degrees_to_wire(degrees):
    return int((int(degrees) + 180) * 10)


def validate_config(cfg):
    required = ("connection", "arm", "infrared", "gripper", "chassis", "vision")
    if any(key not in cfg for key in required):
        raise ValueError("configuration is missing a required section")
    arm = cfg["arm"]
    if not isinstance(arm.get("calibration_verified", False), bool):
        raise ValueError("arm.calibration_verified must be a boolean")
    for key in ("distal_servo_id", "base_servo_id"):
        if arm[key] not in (1, 2, 3):
            raise ValueError(key + " must be an SDK servo ID in [1, 3]")
    if arm["distal_servo_id"] == arm["base_servo_id"]:
        raise ValueError("distal and base servo IDs must differ")
    for key in ("distal_feedback_slot", "base_feedback_slot"):
        if arm[key] not in range(4):
            raise ValueError(key + " must be a DDS feedback slot in [0, 3]")
    if arm["distal_feedback_slot"] == arm["base_feedback_slot"]:
        raise ValueError("distal and base feedback slots must differ")
    for key in ("distal_hold_raw", "base_extended_raw", "base_retracted_raw"):
        value = arm[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2048:
            raise ValueError(key + " must be a DDS encoder value in [0, 2048]")
        if not -180 <= raw_to_sdk_degrees(value) <= 180:
            raise ValueError(key + " cannot be represented by Servo.moveto")
    ir = cfg["infrared"]
    if ir["feedback_slot"] not in range(4):
        raise ValueError("infrared.feedback_slot must be in [0, 3]")
    if ir["threshold_mm"] != 30:
        raise ValueError("this demo requires an infrared threshold of exactly 30 mm")
    if not 2 <= ir["consecutive_samples"] <= 20:
        raise ValueError("infrared.consecutive_samples must be in [2, 20]")
    if not -180 <= cfg["chassis"]["place_turn_degrees"] < 0:
        raise ValueError("place_turn_degrees must be negative for a right turn")
    return cfg


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


class EventLog:
    def __init__(self, path, report):
        self.path = path
        self.report = report
        self.lock = threading.Lock()

    def write(self, stage, **values):
        item = {"time_utc": utc_now(), "stage": stage, **values}
        with self.lock:
            self.report["events"].append(item)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            temporary.replace(self.path)
        print(json.dumps(item, ensure_ascii=False), flush=True)


class Feedback:
    def __init__(self):
        self.condition = threading.Condition()
        self.servo = None
        self.servo_time = 0.0
        self.distance = None
        self.distance_time = 0.0

    def on_servo(self, value):
        raw = copy.deepcopy(value)
        if len(raw) != 3 or any(len(part) != 4 for part in raw):
            return
        with self.condition:
            self.servo = raw
            self.servo_time = time.monotonic()
            self.condition.notify_all()

    def on_distance(self, value):
        raw = copy.deepcopy(value)
        if len(raw) != 4:
            return
        with self.condition:
            self.distance = raw
            self.distance_time = time.monotonic()
            self.condition.notify_all()

    def wait_ready(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while time.monotonic() < deadline:
                if self.servo is not None and self.distance is not None:
                    return
                self.condition.wait(timeout=0.1)
        raise RuntimeError("servo or infrared feedback did not arrive")

    def servo_raw(self, slot, freshness=1.0):
        with self.condition:
            if self.servo is None or time.monotonic() - self.servo_time > freshness:
                raise RuntimeError("servo feedback is missing or stale")
            valid, _speed, angle = self.servo
            if not valid[slot]:
                raise RuntimeError("servo feedback slot {} is offline".format(slot))
            return int(angle[slot])


def yolo_worker(frame_queue, stop_event, model_name, confidence, image_size, log_path):
    """Inference-only subprocess. Its output is deliberately not returned to control."""
    def emit(stage, **values):
        item = {"time_utc": utc_now(), "stage": stage, **values}
        with Path(log_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("YOLO " + json.dumps(item, ensure_ascii=False), flush=True)

    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        emit("model_ready", model=model_name)
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            results = model.predict(frame, conf=confidence, imgsz=image_size, verbose=False)
            detections = []
            for result in results:
                names = result.names
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": round(float(box.conf[0]), 4),
                        "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    })
            emit("detections", count=len(detections), detections=detections)
    except BaseException as exc:
        emit("vision_error", error="{}: {}".format(type(exc).__name__, exc))


class VisionSidecar:
    def __init__(self, bot, cfg, log_path, record):
        self.bot, self.cfg, self.log_path, self.record = bot, cfg, log_path, record
        self.context = mp.get_context("spawn")
        self.manager = None
        self.frames = None
        self.stop_event = self.context.Event()
        self.process = None
        self.thread = None
        self.running = False

    def start(self, camera_module):
        if not self.cfg["enabled"]:
            self.record("vision_disabled")
            return
        # A managed queue has no parent-side feeder thread writing to a killed
        # inference process. This also permits bounded shutdown during model load.
        self.manager = self.context.Manager()
        self.frames = self.manager.Queue(maxsize=1)
        model = self.cfg["model"]
        local_model = ROOT / "models" / model
        model_arg = str(local_model) if local_model.exists() else model
        self.process = self.context.Process(
            target=yolo_worker,
            args=(self.frames, self.stop_event, model_arg, self.cfg["confidence"],
                  self.cfg["image_size"], str(self.log_path)),
            name="yolo-log-worker",
            daemon=True,
        )
        self.process.start()
        try:
            ok = self.bot.camera.start_video_stream(
                display=False, resolution=camera_module.STREAM_720P)
            if ok is False:
                raise RuntimeError("camera rejected video stream request")
        except Exception as exc:
            self.record("vision_camera_error", error=str(exc), control_continues=True)
            self.stop()
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture, name="camera-log-feed", daemon=True)
        self.thread.start()
        self.record("vision_started", model=model_arg, affects_control=False,
                    detection_log=str(self.log_path))

    def _capture(self):
        interval = self.cfg["frame_interval_s"]
        try:
            while not self.stop_event.is_set():
                frame = self.bot.camera.read_cv2_image(strategy="newest", timeout=1.0)
                if frame is not None:
                    try:
                        self.frames.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            pass
                self.stop_event.wait(interval)
        except BaseException as exc:
            self.record("vision_capture_error", error="{}: {}".format(type(exc).__name__, exc),
                        control_continues=True)

    def stop(self):
        self.stop_event.set()
        if self.running:
            try:
                self.bot.camera.stop_video_stream()
            except Exception:
                pass
            self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.frames is not None:
            try:
                self.frames.put_nowait(None)
            except (queue.Full, EOFError, BrokenPipeError):
                pass
        if self.process and self.process.pid is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None
            self.frames = None


class Demo:
    def __init__(self, bot, cfg, feedback, record):
        self.bot, self.cfg, self.feedback, self.record = bot, cfg, feedback, record
        self.arm = cfg["arm"]
        self.distal_locked = False
        self.active_action = None

    def assert_distal(self):
        if not self.distal_locked:
            return
        actual = self.feedback.servo_raw(self.arm["distal_feedback_slot"])
        error = abs(actual - self.arm["distal_hold_raw"])
        if error > self.arm["distal_drift_limit_raw"]:
            raise RuntimeError("distal servo moved: raw error {} exceeds {}".format(
                error, self.arm["distal_drift_limit_raw"]))

    def move_servo(self, servo_id, feedback_slot, target_raw, stage):
        if servo_id == self.arm["distal_servo_id"] and self.distal_locked:
            raise RuntimeError("distal servo is locked; additional position commands are forbidden")
        if servo_id not in (self.arm["distal_servo_id"], self.arm["base_servo_id"]):
            raise RuntimeError("servo ID is outside the configured arm")
        self.record(stage + "_request", servo_id=servo_id, target_raw=target_raw,
                    target_degrees=raw_to_sdk_degrees(target_raw),
                    expected_dds_raw=sdk_degrees_to_raw(raw_to_sdk_degrees(target_raw)),
                    encoded_target_wire=sdk_degrees_to_wire(raw_to_sdk_degrees(target_raw)))
        action = self.bot.servo.moveto(index=servo_id, angle=raw_to_sdk_degrees(target_raw))
        self.active_action = action
        done = action.wait_for_completed(timeout=self.arm["action_timeout_s"])
        if not done or not action.has_succeeded:
            state = getattr(action, "state", "unknown")
            reason = getattr(action, "failure_reason", None)
            self.record(stage + "_action_failed", state=state, failure_reason=str(reason),
                        actual_raw=self.feedback.servo_raw(feedback_slot), waited_to_completion=done)
            raise RuntimeError(stage + ": servo action failed (state={}, reason={}, completed={}). "
                               "If EP is configured as a robotic arm in the App, direct servo "
                               "control is unavailable; configure independent servos for this demo."
                               .format(state, reason, done))
        deadline = time.monotonic() + self.arm["action_timeout_s"]
        while time.monotonic() < deadline:
            actual = self.feedback.servo_raw(feedback_slot)
            if abs(actual - target_raw) <= self.arm["raw_tolerance"]:
                time.sleep(self.arm["settle_seconds"])
                actual = self.feedback.servo_raw(feedback_slot)
                if abs(actual - target_raw) <= self.arm["raw_tolerance"]:
                    self.active_action = None
                    self.record(stage + "_complete", actual_raw=actual)
                    return
            time.sleep(0.05)
        raise RuntimeError(stage + ": servo feedback did not settle at target")

    def lock_initial_distal(self):
        if self.distal_locked:
            raise RuntimeError("distal servo lock may only be issued once")
        self.move_servo(self.arm["distal_servo_id"], self.arm["distal_feedback_slot"],
                        self.arm["distal_hold_raw"], "lock_distal")
        self.distal_locked = True
        self.record("distal_command_guard_enabled", servo_id=self.arm["distal_servo_id"],
                    feedback_slot=self.arm["distal_feedback_slot"], hold_raw=self.arm["distal_hold_raw"])

    def move_base(self, target_raw, stage):
        self.assert_distal()
        self.move_servo(self.arm["base_servo_id"], self.arm["base_feedback_slot"],
                        target_raw, stage)
        self.assert_distal()

    def gripper(self, opened, stage):
        self.assert_distal()
        section = self.cfg["gripper"]
        power = section["open_power" if opened else "close_power"]
        command = self.bot.gripper.open if opened else self.bot.gripper.close
        self.record(stage + "_request", opened=opened, power=power)
        if not command(power=power):
            raise RuntimeError(stage + ": gripper command rejected")
        time.sleep(section["open_seconds" if opened else "close_seconds"])
        if not self.bot.gripper.pause():
            raise RuntimeError(stage + ": gripper pause rejected")
        self.assert_distal()
        self.record(stage + "_complete", opened=opened)

    def wait_for_object(self):
        ir = self.cfg["infrared"]
        slot, threshold = ir["feedback_slot"], ir["threshold_mm"]
        deadline = time.monotonic() + ir["wait_timeout_s"]
        sequence = 0
        last_processed = None
        while time.monotonic() < deadline:
            self.assert_distal()
            with self.feedback.condition:
                if (self.feedback.distance is None or
                        time.monotonic() - self.feedback.distance_time > ir["freshness_timeout_s"]):
                    sequence = 0
                    self.feedback.condition.wait(timeout=0.1)
                    continue
                distance = int(self.feedback.distance[slot])
                measured_at = self.feedback.distance_time
                self.feedback.condition.wait(timeout=0.05)
            if last_processed == measured_at:
                continue
            if (last_processed is not None and
                    measured_at - last_processed > ir["freshness_timeout_s"]):
                sequence = 0
            last_processed = measured_at
            self.record("infrared", slot=slot, distance_mm=distance,
                        threshold_mm=threshold)
            sequence = sequence + 1 if 0 < distance <= threshold else 0
            if sequence >= ir["consecutive_samples"]:
                self.record("grasp_distance_confirmed", distance_mm=distance,
                            consecutive_samples=sequence)
                return
        raise RuntimeError("no object stayed within 30 mm before timeout")

    def turn(self, degrees, stage):
        self.assert_distal()
        chassis = self.cfg["chassis"]
        self.record(stage + "_request", degrees=degrees)
        action = self.bot.chassis.move(x=0, y=0, z=degrees, z_speed=chassis["turn_speed_dps"])
        self.active_action = action
        done = action.wait_for_completed(timeout=chassis["action_timeout_s"])
        if not done or not action.has_succeeded:
            raise RuntimeError(stage + ": chassis action failed or timed out")
        self.active_action = None
        self.assert_distal()
        self.record(stage + "_complete", degrees=degrees)

    def initial_state(self, prefix):
        self.move_base(self.arm["base_extended_raw"], prefix + "_extend")
        self.gripper(True, prefix + "_open_gripper")
        self.assert_distal()

    def run(self):
        self.lock_initial_distal()
        self.initial_state("start")
        self.record("initial_state_ready", arm="extended", gripper="open")
        self.wait_for_object()
        self.gripper(False, "grasp_close")
        self.move_base(self.arm["base_retracted_raw"], "carry_retract")
        turn = self.cfg["chassis"]["place_turn_degrees"]
        self.turn(turn, "turn_to_place")
        self.move_base(self.arm["base_extended_raw"], "place_extend")
        self.gripper(True, "place_release")
        self.move_base(self.arm["base_retracted_raw"], "return_retract")
        self.turn(-turn, "turn_to_start")
        self.initial_state("finish")
        self.record("cycle_complete", arm="extended", gripper="open", heading_degrees=0)

    def emergency_stop(self):
        try:
            self.bot.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        except Exception:
            pass
        try:
            self.bot.servo.pause(index=self.arm["base_servo_id"])
        except Exception:
            pass
        try:
            self.bot.gripper.pause()
        except Exception:
            pass


def configure_network(connection, rm_config):
    if connection.get("local_ip"):
        rm_config.LOCAL_IP_STR = connection["local_ip"]
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((connection["robot_ip"], 20020))
            rm_config.LOCAL_IP_STR = route.getsockname()[0]
    rm_config.ROBOT_IP_STR = connection["robot_ip"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "ir_pick_place.json")
    parser.add_argument("--execute", action="store_true", help="send physical motion commands")
    parser.add_argument("--no-yolo", action="store_true", help="disable the independent YOLO logger")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.no_yolo:
        cfg["vision"]["enabled"] = False
    if not args.execute:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        print("预览：锁定抓夹侧 raw 601 → 外伸/松爪 → 红外≤30 mm → 夹紧 → 内收 → 右转90° → 外伸/松爪 → 内收 → 左转90° → 外伸/松爪")
        print("未连接机器人；实机运行需添加 --execute。")
        return 0

    if not cfg["arm"].get("calibration_verified", False):
        print("提示：舵机映射及外伸/内收位置尚未实机确认；本次按配置执行，"
              "抓夹侧 raw={}，外伸 raw={}，内收 raw={}。".format(
                  cfg["arm"]["distal_hold_raw"], cfg["arm"]["base_extended_raw"],
                  cfg["arm"]["base_retracted_raw"]), flush=True)

    try:
        from robomaster import camera, config as rm_config, led, robot
    except ImportError as exc:
        parser.exit(2, "RoboMaster SDK unavailable: {}\n".format(exc))
    configure_network(cfg["connection"], rm_config)
    output_dir = ROOT / "work" / ("ir-demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output_dir.mkdir(parents=True)
    report_path = output_dir / "run.json"
    report = {"started_utc": utc_now(), "completed": False, "error": None,
              "config": cfg, "events": []}
    event_log = EventLog(report_path, report)
    lock_file = (ROOT / "work" / "real-control.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, "another real-robot control process holds work/real-control.lock\n")

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("operator requested stop")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    bot, demo, vision = robot.Robot(), None, None
    servo_subscribed = distance_subscribed = False
    event_log.write("starting")
    try:
        if not bot.initialize(conn_type=cfg["connection"]["conn_type"], proto_type="udp"):
            raise RuntimeError("SDK initialization failed")
        feedback = Feedback()
        servo_subscribed = bool(bot.servo.sub_servo_info(freq=20, callback=feedback.on_servo))
        distance_subscribed = bool(bot.sensor.sub_distance(freq=20, callback=feedback.on_distance))
        if not servo_subscribed or not distance_subscribed:
            raise RuntimeError("servo or infrared subscription was rejected")
        feedback.wait_ready()
        demo = Demo(bot, cfg, feedback, event_log.write)
        try:
            vision = VisionSidecar(bot, cfg["vision"], output_dir / "detections.jsonl", event_log.write)
            vision.start(camera)
        except Exception as vision_exc:
            event_log.write("vision_start_error", error=str(vision_exc), control_continues=True)
        demo.run()
        report["completed"] = True
        event_log.write("finished_successfully", output_dir=str(output_dir))
        bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
    except BaseException as exc:
        report["completed"] = False
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        if demo:
            demo.emergency_stop()
        try:
            bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=255, g=0, b=0, effect=led.EFFECT_ON)
        except Exception as led_exc:
            event_log.write("error_led_failed", error=str(led_exc))
        event_log.write("error", error=report["error"], automatic_return=False)
    finally:
        if vision:
            try:
                vision.stop()
            except Exception as vision_exc:
                event_log.write("vision_cleanup_error", error=str(vision_exc))
        if distance_subscribed:
            try:
                bot.sensor.unsub_distance()
            except Exception:
                pass
        if servo_subscribed:
            try:
                bot.servo.unsub_servo_info()
            except Exception:
                pass
        try:
            bot.close()
        except Exception:
            pass
        report["finished_utc"] = utc_now()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("运行记录：" + str(report_path), flush=True)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
