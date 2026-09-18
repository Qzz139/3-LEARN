#!/usr/bin/env python3
"""Align one COCO sports ball by chassis yaw, then use the proven IR pick/place."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import ir_pick_place_demo as base

ROOT = base.ROOT


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


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


class Detector:
    def __init__(self, cfg, directory, record):
        from ultralytics import YOLO
        import numpy as np
        self.cfg, self.directory, self.record = cfg, directory, record
        self.path = directory / 'detections.jsonl'
        self.model = YOLO(str(ROOT / 'models' / cfg['vision']['model']))
        record('vision_warming', model=cfg['vision']['model'])
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=cfg['vision']['image_size'], verbose=False)
        record('vision_ready', model_ready=True)
        self.sequence = 0

    def detect(self, camera):
        import cv2
        captured = time.monotonic()
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

    def infrared_confirmed(self):
        ir, sequence, last = self.cfg['infrared'], 0, None
        started = time.monotonic()
        deadline = started + .6
        while time.monotonic() < deadline:
            self.assert_distal()
            with self.feedback.condition:
                stamped = self.feedback.distance_time
                if self.feedback.distance is None or time.monotonic() - stamped > ir['freshness_timeout_s']:
                    sequence = 0
                elif stamped >= started and stamped != last:
                    value = int(self.feedback.distance[ir['feedback_slot']])
                    sequence = sequence + 1 if 0 < value <= ir['threshold_mm'] else 0
                    last = stamped
                    if sequence >= ir['consecutive_samples']:
                        self.record('grasp_distance_confirmed', distance_mm=value, consecutive_samples=sequence)
                        return True
                self.feedback.condition.wait(timeout=.05)
        return False

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
                self.turn(turn, 'align_turn')
                time.sleep(a['settle_seconds'])
                continue
            stable += 1
            self.record('alignment_stable', consecutive_frames=stable)
            if stable >= a['stable_frames'] and self.infrared_confirmed():
                self.record('alignment_and_distance_ready', target=target, heading_offset=self.heading_offset())
                return
            time.sleep(.1)
        raise RuntimeError('target did not become aligned and IR-ready before timeout; no forward drive is performed')

    def run(self):
        base.require_motion_targets(self.cfg)
        self.start_yaw = self.feedback.yaw()
        self.lock_initial_distal()
        self.initial_state('start')
        self.record('initial_state_ready', start_yaw=self.start_yaw)
        self.align_and_wait_for_grasp()
        self.gripper(False, 'grasp_close')
        self.move_base(self.arm['base_retracted_raw'], 'carry_retract')
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
    args = parser.parse_args()
    try:
        cfg = validate_alignment(json.loads(args.config.read_text(encoding='utf-8')))
        base.require_motion_targets(cfg)
        if args.execute and args.observe:
            raise ValueError('choose --execute or --observe')
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not args.execute and not args.observe:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        print('预览：YOLO网球 → 底盘小步对准 → 连续居中且红外≤20mm → 抓取 → 启动朝向右侧90°放置 → 返回；不自动前进。')
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
    bot, demo, camera_started, cleanups = robot.Robot(), None, False, []
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
            ):
                if not subscribe(freq=20, callback=callback):
                    raise RuntimeError('feedback subscription rejected')
                cleanups.append(unsubscribe)
            feedback.wait_ready()
            deadline = time.monotonic() + 5
            while feedback.yaw_value is None and time.monotonic() < deadline:
                time.sleep(.05)
            feedback.yaw()
            demo = AlignDemo(bot, cfg, feedback, log.write, None)
            bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
        if not bot.camera.start_video_stream(display=False, resolution=camera.STREAM_720P):
            raise RuntimeError('camera stream rejected')
        camera_started = True
        log.write('vision_loading', motion_started=False)
        detector = Detector(cfg, directory, log.write)
        if demo:
            demo.detector = detector
            demo.run()
            bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
        else:
            detections, age = detector.detect(bot.camera)
            target = choose_target(detections, cfg['alignment'])
            log.write('observe_result', target=target, age_s=age,
                      proposed_turn=correction_degrees(pixel_error(target, cfg['alignment']['axis_x_ratio']), cfg['alignment']) if target else None,
                      motion_commands_sent=False)
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
