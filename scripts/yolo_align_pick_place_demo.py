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


# 将角度归一化到 [-180, 180)，便于计算跨越 ±180° 时的最短转角。
def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


# 计算两个平面里程计位置之间的直线距离，单位为米。
def position_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


# 先复用基础配置校验，再检查视觉对准、接近速度和停车判据的取值范围。
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
    if p['stop_wheel_rpm'] > 8 or p['stop_encoder_counts'] > 16 or p['stop_stable_seconds'] < .3:
        raise ValueError('wheel stop requires at most 8rpm/16 encoder counts and at least 0.3 seconds')
    return cfg


# 检测框中心相对抓取轴的水平偏差，以图像宽度归一化；正值表示目标在右侧。
def pixel_error(target, axis):
    return (target['xyxy'][0] + target['xyxy'][2]) / 2.0 / target['image_width'] - axis


# 偏差落入容差时无需转向，否则按比例计算并限制单次修正角度。
def correction_degrees(error, alignment):
    if abs(error) <= alignment['tolerance_ratio']:
        return 0.0
    magnitude = min(alignment['max_turn_degrees'],
                    max(alignment['min_turn_degrees'], abs(error) * alignment['gain_degrees']))
    # 目标在画面右侧时底盘顺时针转动（SDK 的 z 为负）；左侧则反向。
    return -math.copysign(magnitude, error)


# 首次选择最接近抓取轴的目标；后续结合类别、中心位移和面积变化维持同一目标。
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


# 生成相对启动朝向的搜索角度：左右交替，并逐步扩大搜索范围。
def search_offsets(alignment):
    step, limit = alignment['search_step_degrees'], alignment['search_limit_degrees']
    offsets, angle = [], step
    while angle <= limit:
        offsets.extend([angle, -angle])
        angle += step
    return offsets


# 扩展基础反馈缓存：线程安全地保存航向、位置、速度及四轮状态。
class Feedback(base.Feedback):
    def __init__(self):
        super().__init__()
        self.yaw_value = None
        self.yaw_time = 0.0
        self.position_value = None
        self.position_time = 0.0
        self.velocity_value = None
        self.velocity_time = 0.0
        self.esc_value = None
        self.esc_time = 0.0

    # 缓存四轮转速、编码器角度和数据包标识，并记录本机单调时钟时间。
    def on_esc(self, data):
        with self.condition:
            self.esc_value = tuple(tuple(int(v) for v in field) for field in data[:3])
            self.esc_time = time.monotonic()
            self.condition.notify_all()

    # 读取轮组反馈快照；数据缺失、过期或字段长度异常时拒绝继续。
    def esc_snapshot(self, freshness):
        with self.condition:
            if (self.esc_value is None or time.monotonic() - self.esc_time > freshness or
                    any(len(field) != 4 for field in self.esc_value)):
                raise RuntimeError('wheel feedback is missing, invalid or stale')
            return self.esc_value, self.esc_time

    # 取 SDK 速度反馈的后半部分，即机体坐标系中的三个速度分量。
    def on_velocity(self, velocity):
        with self.condition:
            self.velocity_value = tuple(float(v) for v in velocity[3:6])
            self.velocity_time = time.monotonic()
            self.condition.notify_all()

    # 检查速度反馈是否及时、数值是否有限，再返回快照及接收时间。
    def velocity_snapshot(self, freshness):
        with self.condition:
            if (self.velocity_value is None or time.monotonic() - self.velocity_time > freshness or
                    not all(math.isfinite(v) for v in self.velocity_value)):
                raise RuntimeError('chassis velocity feedback is missing, invalid or stale')
            return self.velocity_value, self.velocity_time

    # 保存平面位置的 x、y 分量；唤醒等待新反馈的线程。
    def on_position(self, position):
        with self.condition:
            self.position_value = tuple(float(v) for v in position[:2])
            self.position_time = time.monotonic()
            self.condition.notify_all()

    # 读取新鲜的位置反馈，避免依据过期里程计数据控制运动。
    def position(self, freshness):
        with self.condition:
            if (self.position_value is None or time.monotonic() - self.position_time > freshness or
                    not all(math.isfinite(v) for v in self.position_value)):
                raise RuntimeError('chassis position feedback is missing, invalid or stale')
            return self.position_value

    # 保存偏航角，用于转向闭环及相对启动朝向的计算。
    def on_attitude(self, attitude):
        with self.condition:
            self.yaw_value = float(attitude[0])  # SDK 姿态字段依次为偏航、俯仰、横滚。
            self.yaw_time = time.monotonic()
            self.condition.notify_all()

    # 航向反馈超过一秒未更新时中止控制，防止使用过期姿态。
    def yaw(self):
        with self.condition:
            if self.yaw_value is None or time.monotonic() - self.yaw_time > 1.0:
                raise RuntimeError('chassis yaw feedback is missing or stale')
            return self.yaw_value


# 后台持续读取相机最新帧，防止解码队列积压影响后续取图。
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

    # 启动后台采集线程。
    def start(self):
        self.thread.start()

    # 只保留最新图像；取图超时可重试，其他错误交由调用方处理。
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

    # 等待本次调用之后采集的图像，避免使用运动前残留的旧帧。
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

    # 通知采集线程退出，同时唤醒可能正在等待图像的调用方。
    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)


# 负责模型预热、单帧识别，以及检测日志和现场图像的保存。
class Detector:
    def __init__(self, cfg, directory, record):
        from ultralytics import YOLO
        import numpy as np
        self.cfg, self.directory, self.record = cfg, directory, record
        self.path = directory / 'detections.jsonl'
        self.feed = None
        self.model = YOLO(str(ROOT / 'models' / cfg['vision']['model']))
        record('vision_warming', model=cfg['vision']['model'])
        # 使用空白图预热模型，减少首次实际检测的初始化延迟。
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=cfg['vision']['image_size'], verbose=False)
        record('vision_ready', model_ready=True)
        self.sequence = 0

    # 获取图像并执行 YOLO 推理，返回检测列表及图像从采集到推理结束的耗时。
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
        # 同时保存原始图像和标注图；标注图叠加检测框及配置中的抓取轴。
        cv2.imwrite(str(self.directory / 'latest_camera.jpg'), frame)
        annotated = results[0].plot() if results else frame.copy()
        axis = round(width * self.cfg['alignment']['axis_x_ratio'])
        cv2.line(annotated, (axis, 0), (axis, height - 1), (255, 255, 0), 2)
        cv2.imwrite(str(self.directory / 'latest_detections.jpg'), annotated)
        self.record('vision_frame', frame=self.sequence, count=len(detections), age_s=round(age, 3))
        return detections, age


# 在基础机械臂抓放流程上增加视觉对准、低速接近和按记录路径返回。
class AlignDemo(base.Demo):
    def __init__(self, bot, cfg, feedback, record, detector):
        super().__init__(bot, cfg, feedback, record)
        self.detector = detector
        self.start_yaw = None
        self.start_position = None
        self.approach_path = []
        self.travel_m = 0.0

    # 读取新鲜的位置反馈，避免依据过期里程计数据控制运动。
    def position(self):
        return self.feedback.position(self.cfg['approach']['position_freshness_s'])

    # 读取指定红外通道；反馈需足够新且距离为正，否则禁止前进。
    def infrared_value(self):
        ir = self.cfg['infrared']
        with self.feedback.condition:
            if self.feedback.distance is None or time.monotonic() - self.feedback.distance_time > min(ir['freshness_timeout_s'], .25):
                raise RuntimeError('infrared feedback is missing or stale; do not advance')
            value = int(self.feedback.distance[ir['feedback_slot']])
        if value <= 0:
            raise RuntimeError('infrared feedback is invalid; do not advance')
        return value

    # 发送四轮零转速命令，并检查 SDK 是否收到确认。
    def stop_translation(self):
        # 轮速命令有应答确认；用停车定时器替换速度接口定时器，避免重新切回速度模式。
        accepted = self.bot.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=.2)
        self.record('wheel_stop_ack', accepted=bool(accepted))
        if not accepted:
            raise RuntimeError('four-wheel stop command was not acknowledged')

    # 先执行基础急停，再补发四轮停车；停车异常写入运行日志。
    def emergency_stop(self):
        super().emergency_stop()
        try:
            self.stop_translation()
        except Exception as exc:
            self.record('emergency_wheel_stop_error', error=str(exc))

    # 持续检查轮速及编码器变化，达到规定样本数和稳定时长后才确认停稳。
    def wait_stationary(self):
        p, started = self.cfg['approach'], time.monotonic()
        sequence, last_packet, anchor, stable_since = 0, None, None, None
        last_logged, velocity, speeds = started - .1, None, None
        while time.monotonic() < started + p['stop_timeout_s']:
            self.assert_distal()
            velocity, _ = self.feedback.velocity_snapshot(p['position_freshness_s'])
            (speeds, angles, packet), stamped = self.feedback.esc_snapshot(p['position_freshness_s'])
            # 只统计开始等待之后的新数据包，重复反馈不增加稳定样本数。
            if stamped >= started and packet != last_packet:
                last_packet = packet
                if max(abs(v) for v in speeds) > p['stop_wheel_rpm']:
                    anchor, stable_since, sequence = None, None, 0
                else:
                    # 编码器按 32768 回绕，取最短差值判断是否仍有轮子移动。
                    if anchor is None or any(abs((v - ref + 16384) % 32768 - 16384) >
                                             p['stop_encoder_counts'] for v, ref in zip(angles, anchor)):
                        anchor, stable_since, sequence = angles, stamped, 0
                    sequence += 1
                    if sequence >= 3 and stamped - stable_since >= p['stop_stable_seconds']:
                        self.record('translation_stop_confirmed', body_velocity=velocity, wheel_rpm=speeds,
                                    encoder_counts=angles, consecutive_samples=sequence,
                                    stable_seconds=stamped - stable_since)
                        return
                if stamped - last_logged >= .1:
                    self.record('translation_stop_sample', body_velocity=velocity, wheel_rpm=speeds,
                                encoder_counts=angles, consecutive_samples=sequence)
                    last_logged = stamped
            time.sleep(.025)
        self.record('translation_stop_timeout', body_velocity=velocity, wheel_rpm=speeds,
                    consecutive_samples=sequence)
        raise RuntimeError('wheel encoders did not settle after stop')

    # 执行一段平移：前进依据里程计和红外停止，后退依据已记录的路径点停止。
    def translate(self, distance, stage, return_target=None, return_axis=None):
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
        if return_axis is None:
            return_axis = tuple((start[i] - return_target[i]) / target_distance for i in range(2)) if target_distance else None
        elif return_target is not None:
            target_distance = sum((start[i] - return_target[i]) * return_axis[i] for i in range(2))
        peak_distance, peak_position = 0.0, start
        try:
            while time.monotonic() < deadline:
                self.assert_distal()
                current = self.position()
                self.heading_offset()  # 航向反馈过期时禁止继续行驶。
                travelled = position_distance(start, current)
                if return_target is None:
                    progress = travelled
                    if travelled > peak_distance:
                        peak_distance, peak_position = travelled, current
                    value = self.infrared_value()
                    if value <= self.cfg['infrared']['threshold_mm']:
                        self.record('approach_threshold_seen', position=current, distance_mm=value)
                        break
                    if travelled >= distance:
                        break
                else:
                    # 按沿回退方向的投影判断是否到达路径点所在截面；
                    # 横向误差另行检查，避免直线回退时错过路径点附近的小圆形区域。
                    progress = sum((start[i] - current[i]) * return_axis[i] for i in range(2)) if return_axis else 0
                    remaining = target_distance - progress
                    lateral = abs((current[0] - return_target[0]) * return_axis[1] -
                                  (current[1] - return_target[1]) * return_axis[0]) if return_axis else 0
                    if progress < -p['waypoint_tolerance_m'] or lateral > p['center_tolerance_m']:
                        self.record(stage + '_diverged', position=current, progress_m=progress, lateral_error_m=lateral)
                        raise RuntimeError(stage + ': return path diverged from recorded waypoint')
                    if remaining <= p['waypoint_tolerance_m']:
                        break
                if progress > previous_progress + .0005:
                    previous_progress, progress_time = progress, time.monotonic()
                if time.monotonic() - progress_time > p['stall_timeout_s']:
                    raise RuntimeError(stage + ': no odometry progress')
                # 周期续发带 0.2 秒超时的速度命令，主循环卡住时由 SDK 定时停止。
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
            axis = tuple((peak_position[i] - start[i]) / peak_distance for i in range(2)) if peak_distance >= .005 else None
            self.approach_path.append(dict(start=start, end=end, heading=heading, distance=actual,
                                          axis=axis, peak_distance=peak_distance))
            self.record('approach_path_recorded', start_position=start, end_position=end,
                        forward_axis=axis, peak_distance_m=peak_distance)
            self.travel_m += actual
        return actual

    # 在剩余总行程预算内前进一步，并核对实际位移是否超限。
    def approach_step(self):
        p = self.cfg['approach']
        remaining = p['max_travel_m'] - self.travel_m
        if remaining < .001:
            raise RuntimeError('maximum approach travel reached; no grasp')
        self.translate(min(p['step_m'], remaining), 'approach_step')
        if self.travel_m > p['max_travel_m'] + p['waypoint_tolerance_m']:
            raise RuntimeError('actual approach travel exceeded configured limit')

    # 倒序回退接近阶段记录的各段路径，最后核对与启动位置的距离误差。
    def return_to_center(self):
        # 倒序回退每个实测路段，并恢复该段原有朝向，无需转换世界坐标与机体坐标。
        for segment in reversed(self.approach_path):
            self.turn_to_offset(segment['heading'], 'return_path_turn')
            current = self.position()
            distance = position_distance(current, segment['start'])
            if 'axis' in segment:
                axis = segment['axis']
                if axis is None:
                    if distance > self.cfg['approach']['center_tolerance_m']:
                        raise RuntimeError('short forward segment has no reliable return direction')
                    self.record('return_short_segment_skipped', error_m=distance)
                    continue
                delta = tuple(current[i] - segment['start'][i] for i in range(2))
                distance = sum(delta[i] * axis[i] for i in range(2))
                lateral = abs(delta[0] * axis[1] - delta[1] * axis[0])
                if lateral > self.cfg['approach']['center_tolerance_m'] or distance < -self.cfg['approach']['center_tolerance_m']:
                    raise RuntimeError('return waypoint error exceeds configured tolerance')
                if distance > self.cfg['approach']['waypoint_tolerance_m']:
                    self.translate(distance, 'return_path_step', return_target=segment['start'], return_axis=axis)
                else:
                    self.record('return_waypoint_already_reached', remaining_m=distance, lateral_error_m=lateral)
                continue
            if distance > self.cfg['approach']['waypoint_tolerance_m']:
                self.translate(distance, 'return_path_step', return_target=segment['start'])
        error = position_distance(self.position(), self.start_position)
        if error > self.cfg['approach']['center_tolerance_m']:
            self.record('return_center_failed', position=self.position(), error_m=error)
            raise RuntimeError('return center error exceeds configured tolerance')
        self.record('startup_center_reached', position=self.position(), error_m=error)

    # 计算相对启动航向的偏转角，并按配置统一反馈方向与控制方向。
    def heading_offset(self):
        return self.cfg['alignment']['yaw_feedback_sign'] * wrap_degrees(self.feedback.yaw() - self.start_yaw)

    # 利用航向反馈反复修正到指定相对角度，单次最多转动 45°。
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

    # 执行视觉修正转向，并通过实际航向变化确认转向方向和进展。
    def align_turn(self, degrees):
        before = self.heading_offset()
        self.turn(degrees, 'align_turn')
        time.sleep(self.cfg['alignment']['settle_seconds'])
        actual = wrap_degrees(self.heading_offset() - before)
        self.record('align_turn_feedback', requested_degrees=degrees, actual_degrees=actual)
        if actual * degrees <= 0 or abs(actual) < min(.5, abs(degrees) * .25):
            raise RuntimeError('alignment turn completed without actual yaw progress')

    # 用连续的新红外样本消除抖动：返回可抓取、仍偏远或继续等待。
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

    # 循环识别、搜索和对准；连续居中后检查距离，必要时小步接近。
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
            # 推理结果过旧时重新取图，不用滞后画面决定下一步运动。
            if age > a['max_result_age_s']:
                self.record('stale_vision_discarded', age_s=age)
                continue
            target = choose_target(detections, a, previous)
            if target is None:
                stable = 0
                if previous is not None:
                    lost += 1
                    self.record('target_missing', consecutive_frames=lost)
                    # 已锁定目标持续丢失时终止，避免突然改抓另一物体。
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
                    stable = 0  # 每次平移后重新累计居中帧数，所需帧数由 stable_frames 配置。
            time.sleep(.1)
        raise RuntimeError('target did not become aligned and IR-ready before timeout')

    # 完成一次抓放：记录起点、对准接近、抓取内收、退回中心、转向放置并恢复初态。
    def run(self):
        base.require_motion_targets(self.cfg)
        self.lock_initial_distal()
        self.initial_state('start')
        self.wait_stationary()
        self.start_yaw = self.feedback.yaw()
        self.start_position = self.position()
        self.record('initial_state_ready', start_yaw=self.start_yaw, start_position=self.start_position)
        self.align_and_wait_for_grasp()
        self.gripper(False, 'grasp_close')
        self.move_base(self.arm['base_retracted_raw'], 'carry_retract')
        self.return_to_center()
        # 放置方向以启动朝向为基准，避免视觉对准时的累计转向改变最终放置方位。
        self.turn_to_offset(self.cfg['chassis']['place_turn_degrees'], 'turn_to_place')
        self.move_base(self.arm['base_extended_raw'], 'place_extend')
        self.gripper(True, 'place_release')
        self.move_base(self.arm['base_retracted_raw'], 'return_retract')
        self.turn_to_offset(0, 'turn_to_start')
        self.initial_state('finish')
        self.record('cycle_complete', heading_offset=self.heading_offset(), arm='extended', gripper='open')


# 解析参数并管理机器人连接、反馈订阅、视觉采集、执行流程及资源清理。
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
    # 默认只输出配置和流程预览；显式指定执行或观察模式后才连接机器人。
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
    # 非阻塞独占锁：已有控制进程占用机器人时，本次立即退出。
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
            # 按 20 Hz 订阅反馈，并记录对应的取消订阅方法用于退出清理。
            for subscribe, unsubscribe, callback in (
                (bot.servo.sub_servo_info, bot.servo.unsub_servo_info, feedback.on_servo),
                (bot.sensor.sub_distance, bot.sensor.unsub_distance, feedback.on_distance),
                (bot.chassis.sub_attitude, bot.chassis.unsub_attitude, feedback.on_attitude),
                (bot.chassis.sub_position, bot.chassis.unsub_position, feedback.on_position),
                (bot.chassis.sub_velocity, bot.chassis.unsub_velocity, feedback.on_velocity),
                (bot.chassis.sub_esc, bot.chassis.unsub_esc, feedback.on_esc),
            ):
                if not subscribe(freq=20, callback=callback):
                    raise RuntimeError('feedback subscription rejected')
                cleanups.append(unsubscribe)
            feedback.wait_ready()
            deadline = time.monotonic() + 5
            while (feedback.yaw_value is None or feedback.position_value is None or feedback.velocity_value is None or feedback.esc_value is None) and time.monotonic() < deadline:
                time.sleep(.05)
            feedback.yaw()
            feedback.position(cfg['approach']['position_freshness_s'])
            feedback.velocity_snapshot(cfg['approach']['position_freshness_s'])
            feedback.esc_snapshot(cfg['approach']['position_freshness_s'])
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
    # 包含用户中断在内的异常统一记录；执行模式下急停并点亮红灯。
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
        # 按订阅的相反顺序释放反馈资源，即使某项失败也继续清理其余项目。
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
