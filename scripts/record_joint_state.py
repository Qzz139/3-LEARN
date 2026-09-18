#!/usr/bin/env python3
# 关节状态只读记录工具：保留SDK原始反馈，不移动或锁定任何关节。
"""Read EP telemetry without sending arm, servo, chassis or gripper motion commands."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import threading
import time


# 生成记录时间标签；反馈年龄和采样时长均使用不受系统校时影响的monotonic。
def utc_now():
    return datetime.now(timezone.utc).isoformat()


# 线程安全记录舵机原始数组和机械臂末端位置，不推断尚未确认的舵机ID映射。
class Recorder:
    # 初始化最新反馈、各源时间戳及各反馈槽的角度采样历史。
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.latest = {}
        self.received = {}
        self.counts = {"servo": 0, "arm": 0}
        self.angles = {str(i): [] for i in range(4)}
        self.errors = []

    # 校验并复制异步反馈，保存来源和时间；保留原始值，不自动换算成关节角度。
    def receive(self, kind, data):
        # The SDK reuses its telemetry lists. Copy inside the callback.
        raw = copy.deepcopy(data)
        now = time.monotonic()
        with self.lock:
            try:
                if kind == "servo":
                    valid, speed, angle = raw
                    if not all(len(v) == 4 for v in (valid, speed, angle)):
                        raise ValueError("expected three arrays of four entries")
                    if any(v not in (0, 1) for v in valid):
                        raise ValueError("invalid online flags")
                    if not all(isinstance(v, (int, float)) and math.isfinite(v)
                               for v in speed + angle):
                        raise ValueError("non-finite servo data")
                    payload = {"valid": list(valid), "speed_raw": list(speed),
                               "angle_raw": list(angle)}
                    for i, active in enumerate(valid):
                        if active:
                            self.angles[str(i)].append(angle[i])
                else:
                    if len(raw) != 2 or not all(isinstance(v, int) for v in raw):
                        raise ValueError("expected two SDK position integers")
                    payload = {"x_raw_sdk_mm": raw[0], "y_raw_sdk_mm": raw[1]}
                event = {"time_utc": utc_now(), "source": kind, "data": payload}
                self.stream.write(json.dumps(event) + "\n")
                self.stream.flush()
                self.latest[kind] = event
                self.received[kind] = now
                self.counts[kind] += 1
            except (TypeError, ValueError) as exc:
                self.errors.append("{} callback: {}".format(kind, exc))

    # 汇总反馈次数、新鲜度和在线状态，同时给出每个舵机槽的原始角度波动范围。
    def snapshot(self):
        with self.lock:
            ages = {key: time.monotonic() - value for key, value in self.received.items()}
            online = self.latest.get("servo", {}).get("data", {}).get("valid", [])
            complete = (sum(online) >= 2 and all(
                self.counts[k] >= 2 and ages.get(k, float("inf")) <= 1.0
                for k in ("servo", "arm")))
            ranges = {key: {"samples": len(values), "min_raw": min(values),
                            "max_raw": max(values), "span_raw": max(values) - min(values)}
                      for key, values in self.angles.items() if values}
            return {"feedback_complete": complete, "counts": dict(self.counts),
                    "latest": copy.deepcopy(self.latest), "age_seconds": ages,
                    "servo_angle_ranges": ranges, "errors": list(self.errors)}


# 连接并只读订阅两个反馈源，中断或订阅失败时也先保存部分结果再关闭SDK。
def record(bot, directory, seconds, freq, label, conn_type, sdk_version):
    directory.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    cleanup = []
    errors = []
    interrupted = False
    with (directory / "samples.jsonl").open("x", encoding="utf-8") as stream:
        recorder = Recorder(stream)
        try:
            initialized = bot.initialize(conn_type=conn_type)
            if initialized is False:
                raise RuntimeError("SDK connection initialization failed")
            for name, subscribe, unsubscribe in (
                ("servo", bot.servo.sub_servo_info, bot.servo.unsub_servo_info),
                ("arm", bot.robotic_arm.sub_position, bot.robotic_arm.unsub_position),
            ):
                try:
                    ok = subscribe(freq=freq, callback=lambda data, kind=name: recorder.receive(kind, data))
                    if not ok:
                        raise RuntimeError("subscription rejected")
                    cleanup.append(unsubscribe)
                except Exception as exc:
                    errors.append("{}: {}".format(name, exc))
            print("Recording feedback for {} seconds. No motion requested.".format(seconds), flush=True)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            interrupted = True
            errors.append("Interrupted: partial recording retained")
        except Exception as exc:
            errors.append("{}: {}".format(type(exc).__name__, exc))
        finally:
            summary = recorder.snapshot()
            summary.update({"started_utc": started, "finished_utc": utc_now(),
                            "label": label, "sdk_version": sdk_version,
                            "conn_type": conn_type, "frequency_hz": freq,
                            "requested_seconds": seconds, "motion_commands_sent": False,
                            "joint_mapping": {"near_base": None, "far_from_base": None},
                            "notes": ["Servo slots are zero-based DDS array positions, not verified command IDs.",
                                      "Servo angles and speeds are raw SDK values, not calibrated joint degrees.",
                                      "Arm x/y are raw SDK position values: forward/up, not chassis x/y.",
                                      "Feedback is asynchronous; each source has its own timestamp.",
                                      "This records the pose; it does not lock or restore a joint."]})
            summary["errors"].extend(errors)
            summary["status"] = ("complete" if summary["feedback_complete"]
                                 and not summary["errors"] else "partial")
            path = directory / "snapshot.json"
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # Persist first, so a slow SDK shutdown cannot lose the measurement.
            print("{}: {}".format(summary["status"], path.resolve()), flush=True)
            for event in summary["latest"].values():
                print(json.dumps(event, ensure_ascii=False), flush=True)
            for error in summary["errors"]:
                print("ERROR: " + error, flush=True)
            for unsubscribe in reversed(cleanup):
                try:
                    unsubscribe()
                except Exception as exc:
                    print("Cleanup warning: " + str(exc), flush=True)
            try:
                bot.close()
            except Exception as exc:
                print("SDK close warning: " + str(exc), flush=True)
    return 130 if interrupted else (0 if summary["status"] == "complete" else 2)


# 校验采样时长和标签，选择连接地址及输出目录，随后启动只读记录。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current-pose")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--freq", type=int, choices=[5, 10, 20, 50], default=10)
    parser.add_argument("--conn-type", choices=["ap", "sta", "rndis"], default="ap")
    parser.add_argument("--local-ip", help="Jetson IP on the EP network; not the Mac USB SSH address")
    parser.add_argument("--robot-ip", help="Override SDK robot address if needed")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "records" / "joint_states")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
        parser.error("--seconds must be within [1, 120]")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.label):
        parser.error("--label must contain 1-64 letters, digits, underscores or hyphens")
    try:
        import robomaster
        from robomaster import config, robot
    except ImportError as exc:
        parser.exit(2, "RoboMaster SDK unavailable in this Python environment: {}\n".format(exc))
    if args.local_ip:
        config.LOCAL_IP_STR = args.local_ip
    if args.robot_ip:
        config.ROBOT_IP_STR = args.robot_ip
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output_dir.expanduser() / (stamp + "-" + args.label)
    return record(robot.Robot(), directory, args.seconds, args.freq, args.label,
                  args.conn_type, getattr(robomaster, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
