#!/usr/bin/env python3
"""Align and approach one COCO sports ball, then use the proven IR pick/place."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import queue
import signal
import threading
import time

import ir_pick_place_demo as base

ROOT = base.ROOT


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def position_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def validate_alignment(cfg):
    base.validate_config(cfg)
    a = cfg['alignment']
    if a['yaw_feedback_sign'] not in (-1, 1):
        raise ValueError('yaw_feedback_sign must be -1 or 1')
    for key in ('axis_x_ratio', 'tolerance_ratio', 'max_center_jump_ratio'):
        if not 0 < a[key] < 1:
            raise ValueError(key + ' must be between 0 and 1')
    for key in ('gain_degrees', 'min_turn_degrees', 'max_turn_degrees', 'search_step_degrees',
                'search_limit_degrees', 'timeout_s', 'settle_seconds', 'max_result_age_s',
                'heading_tolerance_degrees'):
        if not math.isfinite(a[key]) or a[key] <= 0:
            raise ValueError(key + ' must be positive and finite')
    if not a['min_turn_degrees'] <= a['max_turn_degrees'] <= 10:
        raise ValueError('alignment correction must be at most 10 degrees per step')
    if not a['max_turn_degrees'] <= a['search_limit_degrees'] <= 180:
        raise ValueError('search limit must cover correction step and be at most 180 degrees')
    for key in ('stable_frames', 'lost_frames'):
        if not isinstance(a[key], int) or not 2 <= a[key] <= 10:
            raise ValueError(key + ' must be an integer in [2, 10]')
    if cfg['vision']['model'] != 'yolo26m.pt' or not a['target_class_ids']:
        raise ValueError('this demo uses ordinary yolo26m.pt and configured COCO target classes')
    if cfg['vision']['stream_resolution'] not in ('360p', '540p', '720p'):
        raise ValueError('unsupported video stream resolution')
    p = cfg['approach']
    for key, value in p.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('approach.' + key + ' must be positive and finite')
    if not .001 <= p['step_m'] <= .01 or not p['step_m'] <= p['max_travel_m'] <= .5:
        raise ValueError('approach steps must be 1–10mm and total travel at most 0.5m')
    if max(p['speed_mps'], p['return_speed_mps']) > .05:
        raise ValueError('approach/return speeds must be at most 0.05m/s')
    if p['waypoint_tolerance_m'] >= p['step_m'] or p['position_freshness_s'] > .25:
        raise ValueError('waypoint tolerance must be smaller than step; position freshness at most 0.25s')
    if p['stop_linear_speed_mps'] > .01 or p['stop_yaw_speed_dps'] > 2:
        raise ValueError('stationary speed tolerances must be at most 0.01m/s and 2deg/s')
    return cfg


def pixel_error(target, axis):
    return (target['xyxy'][0] + target['xyxy'][2]) / 2.0 / target['image_width'] - axis


def correction_degrees(error, alignment):
    if abs(error) <= alignment['tolerance_ratio']:
        return 0.0
    magnitude = min(alignment['max_turn_degrees'],
                    max(alignment['min_turn_degrees'], abs(error) * alignment['gain_degrees']))
    # Image right -> clockwise chassis -> SDK negative z; image left -> positive.
    return -math.copysign(magnitude, error)


def choose_target(detections, alignment, previous=None):
    candidates = [d for d in detections if d['class_id'] in alignment['target_class_ids']]
    if previous is None:
        return min(candidates, key=lambda d: (abs(pixel_error(d, alignment['axis_x_ratio'])),
                                              -d['confidence'])) if candidates else None
    pc = [(previous['xyxy'][i] + previous['xyxy'][i + 2]) / 2.0 /
          previous['image_width' if i == 0 else 'image_height'] for i in (0, 1)]
    pa = max(1.0, (previous['xyxy'][2] - previous['xyxy'][0]) *
             (previous['xyxy'][3] - previous['xyxy'][1])) / (previous['image_width'] * previous['image_height'])
    ranked = []
    for d in candidates:
        if d['class_id'] != previous['class_id']:
            continue
        dc = [(d['xyxy'][i] + d['xyxy'][i + 2]) / 2.0 /
              d['image_width' if i == 0 else 'image_height'] for i in (0, 1)]
        jump = math.hypot(dc[0] - pc[0], dc[1] - pc[1])
        area = max(1.0, (d['xyxy'][2] - d['xyxy'][0]) * (d['xyxy'][3] - d['xyxy'][1])) / (d['image_width'] * d['image_height'])
        if jump <= alignment['max_center_jump_ratio'] and .4 <= area / pa <= 2.5:
            ranked.append((jump, -d['confidence'], d))
    return min(ranked, key=lambda item: item[:2])[2] if ranked else None


def search_offsets(alignment):
    step, limit = alignment['search_step_degrees'], alignment['search_limit_degrees']
    offsets, angle = [], step
    while angle <= limit:
        offsets.extend([angle, -angle])
        angle += step
    return offsets


class Feedback(base.Feedback):
    def __init__(self):
        super().__init__()
        self.yaw_value = None
        self.yaw_time = 0.0
        self.position_value = None
        self.position_time = 0.0
        self.velocity_value = None
        self.velocity_time = 0.0

    def on_velocity(self, velocity):
        with self.condition:
            self.velocity_value = tuple(float(v) for v in velocity[3:6])
            self.velocity_time = time.monotonic()
            self.condition.notify_all()

    def velocity_snapshot(self, freshness):
        with self.condition:
            if (self.velocity_value is None or time.monotonic() - self.velocity_time > freshness or
                    not all(math.isfinite(v) for v in self.velocity_value)):
                raise RuntimeError('chassis velocity feedback is missing, invalid or stale')
            return self.velocity_value, self.velocity_time

    def on_position(self, position):
        with self.condition:
            self.position_value = tuple(float(v) for v in position[:2])
            self.position_time = time.monotonic()
            self.condition.notify_all()

    def position(self, freshness):
        with self.condition:
            if (self.position_value is None or time.monotonic() - self.position_time > freshness or
                    not all(math.isfinite(v) for v in self.position_value)):
                raise RuntimeError('chassis position feedback is missing, invalid or stale')
            return self.position_value

    def on_attitude(self, attitude):
        with self.condition:
            self.yaw_value = float(attitude[0])  # Official SDK: yaw, pitch, roll.
            self.yaw_time = time.monotonic()
            self.condition.notify_all()

    def yaw(self):
        with self.condition:
            if self.yaw_value is None or time.monotonic() - self.yaw_time > 1.0:
                raise RuntimeError('chassis yaw feedback is missing or stale')
            return self.yaw_value


class CameraFeed:
    """Continuously drain SDK decoded frames so its H264 receive queue can flow."""
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frame = None
        self.captured = 0.0
        self.error = None
        self.thread = threading.Thread(target=self.capture, daemon=True)

    def start(self):
        self.thread.start()

    def capture(self):
        while not self.stop_event.is_set():
            try:
                frame = self.camera.read_cv2_image(strategy='newest', timeout=1)
                if frame is not None:
                    with self.condition:
                        self.frame, self.captured = frame, time.monotonic()
                        self.condition.notify_all()
            except queue.Empty:
                continue
            except Exception as exc:
                with self.condition:
                    self.error = str(exc)
                    self.condition.notify_all()
                return

    def read_latest(self, timeout=3):
        requested = time.monotonic()
        with self.condition:
            while self.frame is None or self.captured < requested:
                if self.error or self.stop_event.is_set():
                    raise RuntimeError('camera feed stopped: ' + str(self.error))
                remaining = requested + timeout - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('camera feed returned no fresh image')
                self.condition.wait(timeout=remaining)
            return self.frame, self.captured

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)


class Detector:
    def __init__(self, cfg, directory, record):
        from ultralytics import YOLO
        import numpy as np
        self.cfg, self.directory, self.record = cfg, directory, record
        self.path = directory / 'detections.jsonl'
        self.feed = None
        self.model = YOLO(str(ROOT / 'models' / cfg['vision']['model']))
        record('vision_warming', model=cfg['vision']['model'])
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=cfg['vision']['image_size'], verbose=False)
        record('vision_ready', model_ready=True)
        self.sequence = 0

    def detect(self, camera):
        import cv2
        captured = time.monotonic()
        if self.feed:
            frame, captured = self.feed.read_latest()
        else:
            frame = camera.read_cv2_image(strategy='newest', timeout=3)
        if frame is None:
            raise RuntimeError('camera returned no image')
        self.sequence += 1
        height, width = frame.shape[:2]
        results = self.model.predict(frame, conf=self.cfg['vision']['confidence'],
                                     imgsz=self.cfg['vision']['image_size'], verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                detections.append({'class_id': cls, 'class_name': result.names[cls],
                                   'confidence': float(box.conf[0]),
                                   'xyxy': [float(v) for v in box.xyxy[0].tolist()],
                                   'image_width': width, 'image_height': height})
        age = time.monotonic() - captured
        event = {'time_utc': base.utc_now(), 'frame': self.sequence, 'frame_age_s': age,
                 'count': len(detections), 'detections': detections}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        # Keep the raw frame and an image with detections plus the configured grab axis.
        cv2.imwrite(str(self.directory / 'latest_camera.jpg'), frame)
        annotated = results[0].plot() if results else frame.copy()
        axis = round(width * self.cfg['alignment']['axis_x_ratio'])
        cv2.line(annotated, (axis, 0), (axis, height - 1), (255, 255, 0), 2)
        cv2.imwrite(str(self.directory / 'latest_detections.jpg'), annotated)
        self.record('vision_frame', frame=self.sequence, count=len(detections), age_s=round(age, 3))
        return detections, age


class AlignDemo(base.Demo):
    def __init__(self, bot, cfg, feedback, record, detector):
        super().__init__(bot, cfg, feedback, record)
        self.detector = detector
        self.start_yaw = None
        self.start_position = None
        self.approach_path = []
        self.travel_m = 0.0

    def position(self):
        return self.feedback.position(self.cfg['approach']['position_freshness_s'])

    def infrared_value(self):
        ir = self.cfg['infrared']
        with self.feedback.condition:
            if self.feedback.distance is None or time.monotonic() - self.feedback.distance_time > min(ir['freshness_timeout_s'], .25):
                raise RuntimeError('infrared feedback is missing or stale; do not advance')
            value = int(self.feedback.distance[ir['feedback_slot']])
        if value <= 0:
            raise RuntimeError('infrared feedback is invalid; do not advance')
        return value

    def stop_translation(self):
        # ProtoChassisSpeedMode is PUSH/no-ACK. Official SDK returns False even
        # after sending; verify physical progress/stop using position instead.
        self.bot.chassis.drive_speed(x=0, y=0, z=0, timeout=.2)

    def wait_stationary(self):
        p, started = self.cfg['approach'], time.monotonic()
        sequence, last = 0, None
        while time.monotonic() < started + p['stop_timeout_s']:
            self.assert_distal()
            velocity, stamped = self.feedback.velocity_snapshot(p['position_freshness_s'])
            if stamped >= started and stamped != last:
                sequence = sequence + 1 if (math.hypot(*velocity[:2]) <= p['stop_linear_speed_mps'] and
                                             abs(velocity[2]) <= p['stop_yaw_speed_dps']) else 0
                last = stamped
                if sequence >= 3:
                    self.record('translation_stop_confirmed', body_velocity=velocity, consecutive_samples=sequence)
                    return
            time.sleep(.025)
        raise RuntimeError('chassis velocity did not settle after stop')

    def translate(self, distance, stage, return_target=None):
        """Odometry controls each segment; the SDK timer stops a stalled main loop."""
        p = self.cfg['approach']
        self.assert_distal()
        start, heading = self.position(), self.heading_offset()
        deadline = time.monotonic() + p['segment_timeout_s']
        direction = 1 if return_target is None else -1
        speed = p['speed_mps'] if direction > 0 else p['return_speed_mps']
        self.record(stage + '_request', distance_m=distance, speed_mps=direction * speed,
                    start_position=start, heading_offset=heading, return_target=return_target)
        previous_progress, progress_time = 0.0, time.monotonic()
        target_distance = position_distance(start, return_target) if return_target is not None else None
        try:
            while time.monotonic() < deadline:
                self.assert_distal()
                current = self.position()
                self.heading_offset()  # Refuse to drive with stale yaw feedback.
                travelled = position_distance(start, current)
                if return_target is None:
                    progress = travelled
                    value = self.infrared_value()
                    if value <= self.cfg['infrared']['threshold_mm']:
                        self.record('approach_threshold_seen', position=current, distance_mm=value)
                        break
                    if travelled >= distance:
                        break
                else:
                    remaining = position_distance(current, return_target)
                    progress = target_distance - remaining
                    if remaining <= p['waypoint_tolerance_m']:
                        break
                    if remaining > target_distance + p['waypoint_tolerance_m'] or travelled > distance + p['waypoint_tolerance_m']:
                        raise RuntimeError(stage + ': return path diverged from recorded waypoint')
                if progress > previous_progress + .0005:
                    previous_progress, progress_time = progress, time.monotonic()
                if time.monotonic() - progress_time > p['stall_timeout_s']:
                    raise RuntimeError(stage + ': no odometry progress')
                self.bot.chassis.drive_speed(x=direction * speed, y=0, z=0, timeout=.2)
                time.sleep(.025)
            else:
                raise RuntimeError(stage + ': translation timed out')
        finally:
            self.stop_translation()
        time.sleep(p['settle_seconds'])
        self.wait_stationary()
        end = self.position()
        actual = position_distance(start, end)
        self.assert_distal()
        self.record(stage + '_complete', end_position=end, actual_distance_m=actual)
        if return_target is None and actual > .0001:
            self.approach_path.append(dict(start=start, end=end, heading=heading, distance=actual))
            self.travel_m += actual
        return actual

    def approach_step(self):
        p = self.cfg['approach']
        remaining = p['max_travel_m'] - self.travel_m
        if remaining < .001:
            raise RuntimeError('maximum approach travel reached; no grasp')
        self.translate(min(p['step_m'], remaining), 'approach_step')
        if self.travel_m > p['max_travel_m'] + p['waypoint_tolerance_m']:
            raise RuntimeError('actual approach travel exceeded configured limit')

    def return_to_center(self):
        # Revisit each measured segment in reverse, preserving its original heading.
        # This needs no unverified world/body coordinate conversion.
        for segment in reversed(self.approach_path):
            self.turn_to_offset(segment['heading'], 'return_path_turn')
            distance = position_distance(self.position(), segment['start'])
            if distance > self.cfg['approach']['waypoint_tolerance_m']:
                self.translate(distance, 'return_path_step', return_target=segment['start'])
        error = position_distance(self.position(), self.start_position)
        if error > self.cfg['approach']['center_tolerance_m']:
            raise RuntimeError('return center error exceeds configured tolerance')
        self.record('startup_center_reached', position=self.position(), error_m=error)

    def heading_offset(self):
        return self.cfg['alignment']['yaw_feedback_sign'] * wrap_degrees(self.feedback.yaw() - self.start_yaw)

    def turn_to_offset(self, target, stage):
        tolerance = self.cfg['alignment']['heading_tolerance_degrees']
        for _ in range(8):
            error = wrap_degrees(target - self.heading_offset())
            if abs(error) <= tolerance:
                self.record(stage + '_heading_reached', offset=self.heading_offset(), target=target)
                return
            self.turn(max(-45, min(45, error)), stage)
            time.sleep(self.cfg['alignment']['settle_seconds'])
        raise RuntimeError(stage + ': yaw feedback did not reach target heading')

    def align_turn(self, degrees):
        before = self.heading_offset()
        self.turn(degrees, 'align_turn')
        time.sleep(self.cfg['alignment']['settle_seconds'])
        actual = wrap_degrees(self.heading_offset() - before)
        self.record('align_turn_feedback', requested_degrees=degrees, actual_degrees=actual)
        if actual * degrees <= 0 or abs(actual) < min(.5, abs(degrees) * .25):
            raise RuntimeError('alignment turn completed without actual yaw progress')

    def infrared_state(self):
        ir, sequence, far_sequence, last = self.cfg['infrared'], 0, 0, None
        started = time.monotonic()
        deadline = started + .6
        while time.monotonic() < deadline:
            self.assert_distal()
            with self.feedback.condition:
                stamped = self.feedback.distance_time
                if stamped >= started and stamped != last:
                    value = self.infrared_value()
                    near = value <= ir['threshold_mm']
                    sequence = sequence + 1 if near else 0
                    far_sequence = far_sequence + 1 if not near else 0
                    last = stamped
                    if sequence >= ir['consecutive_samples']:
                        self.record('grasp_distance_confirmed', distance_mm=value, consecutive_samples=sequence)
                        return 'ready'
                    if far_sequence >= ir['consecutive_samples']:
                        self.record('approach_distance_pending', distance_mm=value, consecutive_samples=far_sequence)
                        return 'far'
                self.feedback.condition.wait(timeout=.05)
        self.infrared_value()
        return 'wait'

    def align_and_wait_for_grasp(self):
        a, previous, stable, lost = self.cfg['alignment'], None, 0, 0
        offsets = iter(search_offsets(a))
        deadline = time.monotonic() + a['timeout_s']
        while time.monotonic() < deadline:
            self.assert_distal()
            detections, age = self.detector.detect(self.bot.camera)
            self.assert_distal()
            if time.monotonic() >= deadline:
                break
            if age > a['max_result_age_s']:
                self.record('stale_vision_discarded', age_s=age)
                continue
            target = choose_target(detections, a, previous)
            if target is None:
                stable = 0
                if previous is not None:
                    lost += 1
                    self.record('target_missing', consecutive_frames=lost)
                    if lost >= a['lost_frames']:
                        raise RuntimeError('locked target lost; stop instead of switching objects')
                    continue
                offset = next(offsets, None)
                if offset is None:
                    raise RuntimeError('no target found within configured yaw search range')
                self.record('search_target', offset=offset)
                self.turn_to_offset(offset, 'search_turn')
                continue
            previous, lost = target, 0
            error = pixel_error(target, a['axis_x_ratio'])
            turn = correction_degrees(error, a)
            self.record('target_observed', target=target, pixel_error_ratio=error,
                        correction_degrees=turn, heading_offset=self.heading_offset())
            if turn:
                stable = 0
                if abs(self.heading_offset() + turn) > a['search_limit_degrees']:
                    raise RuntimeError('target requires a turn beyond configured search range')
                self.align_turn(turn)
                continue
            stable += 1
            self.record('alignment_stable', consecutive_frames=stable)
            if stable >= a['stable_frames']:
                distance_state = self.infrared_state()
                if distance_state == 'ready':
                    self.record('alignment_and_distance_ready', target=target, heading_offset=self.heading_offset())
                    return
                if distance_state == 'far':
                    self.approach_step()
                    stable = 0  # Require three new centered images after every translation.
            time.sleep(.1)
        raise RuntimeError('target did not become aligned and IR-ready before timeout')

    def run(self):
        base.require_motion_targets(self.cfg)
        self.start_yaw = self.feedback.yaw()
        self.start_position = self.position()
        self.lock_initial_distal()
        self.initial_state('start')
        self.record('initial_state_ready', start_yaw=self.start_yaw, start_position=self.start_position)
        self.align_and_wait_for_grasp()
        self.gripper(False, 'grasp_close')
        self.move_base(self.arm['base_retracted_raw'], 'carry_retract')
        self.return_to_center()
        # Placement is anchored to startup, independent of how much alignment turned.
        self.turn_to_offset(self.cfg['chassis']['place_turn_degrees'], 'turn_to_place')
        self.move_base(self.arm['base_extended_raw'], 'place_extend')
        self.gripper(True, 'place_release')
        self.move_base(self.arm['base_retracted_raw'], 'return_retract')
        self.turn_to_offset(0, 'turn_to_start')
        self.initial_state('finish')
        self.record('cycle_complete', heading_offset=self.heading_offset(), arm='extended', gripper='open')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/yolo_align_pick_place.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--observe', action='store_true', help='camera and YOLO only; no motion or LED commands')
    parser.add_argument('--observe-frames', type=int, default=1, help='number of stationary read-only detections')
    args = parser.parse_args()
    try:
        cfg = validate_alignment(json.loads(args.config.read_text(encoding='utf-8')))
        base.require_motion_targets(cfg)
        if args.execute and args.observe:
            raise ValueError('choose --execute or --observe')
        if not 1 <= args.observe_frames <= 100 or (args.observe_frames != 1 and not args.observe):
            raise ValueError('--observe-frames must be 1–100 and requires --observe')
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not args.execute and not args.observe:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        print('预览：YOLO网球 → 小步转向对准 → 低速约{:g}mm一步接近 → 居中且红外≤20mm → 抓取内收 → 退回起始中心 → 启动朝向右侧90°放置 → 返回。'.format(cfg['approach']['step_m'] * 1000))
        return 0
    from robomaster import camera, config, led, robot
    directory = ROOT / 'work' / ('yolo-align-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True)
    report = {'config': cfg, 'events': [], 'completed': False, 'observe_only': args.observe}
    log = base.EventLog(directory / 'run.json', report)
    lock = (ROOT / 'work/real-control.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(*_):
        raise KeyboardInterrupt('operator stop')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    bot, demo, camera_started, cleanups, feed = robot.Robot(), None, False, [], None
    try:
        base.configure_network(cfg['connection'], config)
        if not bot.initialize(conn_type=cfg['connection']['conn_type'], proto_type='udp'):
            raise RuntimeError('SDK initialize failed')
        if args.execute:
            feedback = Feedback()
            for subscribe, unsubscribe, callback in (
                (bot.servo.sub_servo_info, bot.servo.unsub_servo_info, feedback.on_servo),
                (bot.sensor.sub_distance, bot.sensor.unsub_distance, feedback.on_distance),
                (bot.chassis.sub_attitude, bot.chassis.unsub_attitude, feedback.on_attitude),
                (bot.chassis.sub_position, bot.chassis.unsub_position, feedback.on_position),
                (bot.chassis.sub_velocity, bot.chassis.unsub_velocity, feedback.on_velocity),
            ):
                if not subscribe(freq=20, callback=callback):
                    raise RuntimeError('feedback subscription rejected')
                cleanups.append(unsubscribe)
            feedback.wait_ready()
            deadline = time.monotonic() + 5
            while (feedback.yaw_value is None or feedback.position_value is None or feedback.velocity_value is None) and time.monotonic() < deadline:
                time.sleep(.05)
            feedback.yaw()
            feedback.position(cfg['approach']['position_freshness_s'])
            feedback.velocity_snapshot(cfg['approach']['position_freshness_s'])
            demo = AlignDemo(bot, cfg, feedback, log.write, None)
            bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
        log.write('vision_loading', motion_started=False)
        detector = Detector(cfg, directory, log.write)
        if not bot.camera.start_video_stream(display=False, resolution=cfg['vision']['stream_resolution']):
            raise RuntimeError('camera stream rejected')
        camera_started = True
        feed = CameraFeed(bot.camera)
        feed.start()
        detector.feed = feed
        time.sleep(1.0)
        log.write('camera_ready', resolution=cfg['vision']['stream_resolution'])
        if demo:
            demo.detector = detector
            demo.run()
            bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
        else:
            for _ in range(args.observe_frames):
                detections, age = detector.detect(bot.camera)
                target = choose_target(detections, cfg['alignment'])
                log.write('observe_result', target=target, age_s=age,
                          proposed_turn=correction_degrees(pixel_error(target, cfg['alignment']['axis_x_ratio']), cfg['alignment']) if target else None,
                          motion_commands_sent=False)
                time.sleep(cfg['vision']['frame_interval_s'])
        report['completed'] = True
        log.write('finished_successfully', directory=str(directory))
    except BaseException as exc:
        report['error'] = '{}: {}'.format(type(exc).__name__, exc)
        if demo:
            demo.emergency_stop()
        if args.execute:
            try:
                bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=255, g=0, b=0, effect=led.EFFECT_ON)
            except Exception:
                pass
        log.write('error', error=report['error'], automatic_return=False)
    finally:
        if feed:
            feed.stop()
        if camera_started:
            try:
                bot.camera.stop_video_stream()
            except Exception:
                pass
        for unsubscribe in reversed(cleanups):
            try:
                unsubscribe()
            except Exception:
                pass
        bot.close()
        (directory / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('运行记录：' + str(directory))
    return 0 if report['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
