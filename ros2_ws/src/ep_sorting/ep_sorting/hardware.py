"""One EP SDK connection shared by the camera publisher and sorting Action server."""
import fcntl
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from ep_sorting_interfaces.action import SortObjects

from .legacy import SHARE, align, base, sorting, SortingStateMachine
from .vision_codec import from_detection_array


class HardwareSession:
    """Own the SDK connection; never issue motion during construction."""

    def __init__(self, cfg, lock_path):
        from robomaster import config, robot

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock_path.open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.bot = robot.Robot()
        self.feedback = align.Feedback()
        self.cleanups = []
        self.feed = None
        self.camera_started = False
        try:
            base.configure_network(cfg['connection'], config)
            if not self.bot.initialize(conn_type=cfg['connection']['conn_type'], proto_type='udp'):
                raise RuntimeError('SDK initialize failed')
            for subscribe, unsubscribe, callback in (
                (self.bot.servo.sub_servo_info, self.bot.servo.unsub_servo_info, self.feedback.on_servo),
                (self.bot.sensor.sub_distance, self.bot.sensor.unsub_distance, self.feedback.on_distance),
                (self.bot.chassis.sub_attitude, self.bot.chassis.unsub_attitude, self.feedback.on_attitude),
                (self.bot.chassis.sub_position, self.bot.chassis.unsub_position, self.feedback.on_position),
                (self.bot.chassis.sub_velocity, self.bot.chassis.unsub_velocity, self.feedback.on_velocity),
                (self.bot.chassis.sub_esc, self.bot.chassis.unsub_esc, self.feedback.on_esc),
            ):
                if not subscribe(freq=20, callback=callback):
                    raise RuntimeError('feedback subscription rejected')
                self.cleanups.append(unsubscribe)
            self.feedback.wait_ready()
            deadline = time.monotonic() + 5
            while (self.feedback.yaw_value is None or self.feedback.position_value is None or
                   self.feedback.velocity_value is None or self.feedback.esc_value is None) and time.monotonic() < deadline:
                time.sleep(.05)
            self.feedback.yaw()
            freshness = cfg['approach']['position_freshness_s']
            self.feedback.position(freshness)
            self.feedback.velocity_snapshot(freshness)
            self.feedback.esc_snapshot(freshness)
            if not self.bot.camera.start_video_stream(display=False, resolution=cfg['vision']['stream_resolution']):
                raise RuntimeError('camera stream rejected')
            self.camera_started = True
            self.feed = align.CameraFeed(self.bot.camera)
            self.feed.start()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.feed:
            self.feed.stop()
            self.feed = None
        if self.camera_started:
            try:
                self.bot.camera.stop_video_stream()
            except Exception:
                pass
            self.camera_started = False
        for unsubscribe in reversed(self.cleanups):
            try:
                unsubscribe()
            except Exception:
                pass
        self.cleanups.clear()
        try:
            self.bot.close()
        finally:
            self.lock.close()


class CameraNode(Node):
    def __init__(self, session):
        super().__init__('ep_camera')
        self.session = session
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(Image, '/ep/camera/image_raw', qos)
        self.last_capture = 0.0
        self.timer = self.create_timer(.1, self.publish_frame)

    def publish_frame(self):
        with self.session.feed.condition:
            frame = self.session.feed.frame
            captured = self.session.feed.captured
            error = self.session.feed.error
        if error:
            self.get_logger().error('camera feed stopped: ' + error)
            self.timer.cancel()
            return
        if frame is None or captured <= self.last_capture:
            return
        self.last_capture = captured
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'ep_camera'
        message.height, message.width = height, width
        message.encoding = 'bgr8'
        message.is_bigendian = False
        message.step = width * 3
        message.data = frame.tobytes()
        self.publisher.publish(message)


class ROSDetectorProxy:
    """Present fresh ROS detections through the existing demo's detect() API."""

    def __init__(self, timeout=3.0):
        self.condition = threading.Condition()
        self.latest = None
        self.last_received_mono = 0.0
        self.width = self.height = 0
        self.timeout = timeout
        self.record = None

    def on_image(self, message):
        with self.condition:
            self.width, self.height = message.width, message.height

    def on_detections(self, message):
        with self.condition:
            self.latest = message
            self.last_received_mono = time.monotonic()
            self.condition.notify_all()

    def detect(self, _camera):
        requested_ns = time.time_ns()
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while True:
                message = self.latest
                stamp_ns = (message.header.stamp.sec * 1_000_000_000 +
                            message.header.stamp.nanosec) if message is not None else 0
                if message is not None and stamp_ns >= requested_ns and self.width and self.height:
                    detections = from_detection_array(message, self.width, self.height)
                    age = max(0.0, (time.time_ns() - stamp_ns) / 1e9)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('no fresh ROS detection received')
                self.condition.wait(timeout=remaining)
        if self.record:
            self.record('vision_detections', detections=detections, age_s=age)
        return detections, age


class SortingActionNode(Node):
    def __init__(self, session, cfg, state_file, log_root):
        super().__init__('ep_arm_controller')
        self.session, self.cfg = session, cfg
        self.state_file, self.log_root = Path(state_file), Path(log_root)
        self.detector = ROSDetectorProxy()
        self.active = False
        self.active_demo = None
        self.stopping = False
        self.goal_lock = threading.Lock()
        self.group = ReentrantCallbackGroup()
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/ep/camera/image_raw', self.detector.on_image,
                                 qos, callback_group=self.group)
        self.create_subscription(Detection2DArray, '/ep/detections', self.detector.on_detections,
                                 10, callback_group=self.group)
        self.server = ActionServer(self, SortObjects, '/ep/sort_objects', self.execute,
                                   goal_callback=self.on_goal, cancel_callback=self.on_cancel,
                                   callback_group=self.group)

    def on_goal(self, request):
        with self.goal_lock:
            if (not request.execute or self.active or
                    time.monotonic() - self.detector.last_received_mono > 2.0):
                return GoalResponse.REJECT
            self.active = True
            return GoalResponse.ACCEPT

    def on_cancel(self, _goal):
        return CancelResponse.ACCEPT

    def execute(self, goal):
        directory = self.log_root / ('ros2-sort-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except BaseException:
            self.active = False
            raise
        report = {'config': self.cfg, 'events': [], 'completed': False}
        log = base.EventLog(directory / 'run.json', report)
        result = SortObjects.Result()
        result.log_directory = str(directory)
        counts = {32: 0, 39: 0}
        demo = None

        def record(stage, **values):
            log.write(stage, **values)
            if stage == 'sorting_item_placed':
                counts.update({int(k): v for k, v in values['counts'].items()})
            if stage in ('sorting_state_transition', 'sorting_item_placed', 'sorting_complete'):
                feedback = SortObjects.Feedback()
                feedback.stage = stage
                feedback.state = values.get('target', stage)
                feedback.placed = sum(counts.values())
                goal.publish_feedback(feedback)
            if (goal.is_cancel_requested or self.stopping) and stage not in ('error', 'sorting_state_transition'):
                raise RuntimeError('sorting goal canceled')

        try:
            machine = SortingStateMachine.from_file(self.state_file)
            demo = sorting.SortingDemo(self.session.bot, self.cfg, self.session.feedback,
                                       record, self.detector, state_machine=machine)
            self.active_demo = demo
            self.detector.record = record
            demo.run()
            from robomaster import led
            self.session.bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=0, g=0, b=0,
                                         effect=led.EFFECT_OFF)
            report['completed'] = True
            log.write('finished_successfully', directory=str(directory))
            result.completed = True
            result.all_six_placed = sum(counts.values()) == 6
            goal.succeed()
        except BaseException as exc:
            result.error = '{}: {}'.format(type(exc).__name__, exc)
            report['error'] = result.error
            if demo:
                try:
                    demo.emergency_stop()
                except Exception as stop_error:
                    self.get_logger().error('emergency stop failed: {}'.format(stop_error))
            try:
                from robomaster import led
                self.session.bot.led.set_led(comp=led.COMP_BOTTOM_ALL, r=255, g=0, b=0,
                                             effect=led.EFFECT_ON)
            except Exception:
                pass
            log.write('error', error=result.error, automatic_return=False)
            if goal.is_cancel_requested:
                goal.canceled()
            else:
                goal.abort()
        finally:
            result.balls_placed, result.bottles_placed = counts[32], counts[39]
            (directory / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
            self.active_demo = None
            self.detector.record = None
            self.active = False
        return result


def main(args=None):
    rclpy.init(args=args)
    session = None
    executor = MultiThreadedExecutor(num_threads=4)
    nodes = []
    try:
        options = Node('ep_hardware_options')
        nodes.append(options)
        options.declare_parameter('config_path', str(SHARE / 'config/object_sorting.json'))
        options.declare_parameter('state_machine_path', str(SHARE / 'config/sorting_state_machine.yaml'))
        options.declare_parameter('log_root', str(Path.home() / 'projects/3-LEARN/work'))
        cfg = sorting.validate_sorting(json.loads(Path(options.get_parameter('config_path').value).read_text()))
        sorting.require_placement_slots(cfg)
        base.require_motion_targets(cfg)
        state_file = options.get_parameter('state_machine_path').value
        SortingStateMachine.from_file(state_file)
        log_root = Path(options.get_parameter('log_root').value).expanduser()
        session = HardwareSession(cfg, log_root / 'real-control.lock')
        nodes.extend([CameraNode(session), SortingActionNode(session, cfg, state_file, log_root)])
        for node in nodes:
            executor.add_node(node)
        executor.spin()
    finally:
        if session and len(nodes) >= 3 and nodes[-1].active_demo:
            nodes[-1].stopping = True
            try:
                nodes[-1].active_demo.emergency_stop()
            except Exception:
                pass
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        if session:
            session.close()
        rclpy.shutdown()
