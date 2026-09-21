"""Gazebo-only adapter: YOLO detections and simulated sensors drive a sorting run."""
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vision_msgs.msg import Detection2DArray

from ep_sorting.vision_codec import from_detection_array
from .geometry import PICK_HEADINGS, PICK_OBJECTS, PLACE_HEADINGS, limited_turn, nearest_heading
from .geometry import wrap_radians, yaw_from_quaternion


class SimSorter(Node):
    def __init__(self):
        super().__init__('ep_sim_sorter')
        self.declare_parameter('execute', False)
        self.declare_parameter('log_root', str(Path.home() / 'projects/3-LEARN/work'))
        self.declare_parameter('grasp_range_m', 0.024)
        self.cmd = self.create_publisher(Twist, '/ep/cmd_vel', 10)
        self.joints = self.create_publisher(JointTrajectory, '/ep/set_joint_trajectory', 10)
        sensor_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/ep/odom', self.on_odom, 10)
        self.create_subscription(Image, '/ep/camera/image_raw', self.on_image, sensor_qos)
        self.create_subscription(Range, '/ep/ir/range', self.on_range, sensor_qos)
        self.create_subscription(Detection2DArray, '/ep/detections', self.on_detection, 10)
        self.grip_clients = {}
        self.grip_subscriptions = []
        for name, _class_id in PICK_OBJECTS.values():
            self.grip_clients[name] = self.create_client(SetBool, '/ep/grip/' + name + '/switch')
            self.grip_subscriptions.append(self.create_subscription(
                Bool, '/ep/grip/' + name + '/grasping',
                lambda msg, object_name=name: self.on_grasping(object_name, msg), 10))
        self.create_service(Trigger, '/sim/start', self.on_start)
        self.create_service(Trigger, '/sim/stop', self.on_stop)
        self.state = 'idle'
        self.state_since = time.monotonic()
        self.run_dir = None
        self.log_file = None
        self.pose = None
        self.home = None
        self.home_yaw = None
        self.image_width = 960
        self.last_image_at = 0.0
        self.range_m = math.inf
        self.last_range_at = 0.0
        self.detections = []
        self.last_detection_at = 0.0
        self.grasping = False
        self.last_grasp_at = 0.0
        self.grip_future = None
        self.pending = list(PICK_HEADINGS)
        self.counts = {'ball': 0, 'bottle': 0}
        self.pick_heading = None
        self.pick_object = None
        self.target_class = None
        self.scan_index = 0
        self.heading_seen = False
        self.heading_search_started = 0.0
        self.centered_frames = 0
        self.close_samples = 0
        self.travel_start = None
        self.target_yaw = None
        self.requested = bool(self.get_parameter('execute').value)
        self.create_timer(0.1, self.tick)
        self.get_logger().info('Gazebo ready; call /sim/start to begin the sorting run.')

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))

    def on_image(self, msg):
        self.image_width = msg.width
        self.last_image_at = time.monotonic()

    def on_range(self, msg):
        self.range_m = msg.range
        self.last_range_at = time.monotonic()

    def on_detection(self, msg):
        self.detections = from_detection_array(msg, self.image_width, 540)
        self.last_detection_at = time.monotonic()

    def on_grasping(self, object_name, msg):
        if object_name == self.pick_object:
            self.grasping = bool(msg.data)
            self.last_grasp_at = time.monotonic()

    def on_start(self, _request, response):
        response.success = self.state == 'idle' and not self.requested
        response.message = 'starting' if response.success else 'already started or finished; relaunch Gazebo'
        if response.success:
            self.requested = True
        return response

    def on_stop(self, _request, response):
        response.success = self.state not in ('idle', 'done', 'stopped')
        response.message = 'stopped' if response.success else 'no active run'
        if response.success:
            self.stop()
            self.transition('stopped', reason='operator_stop')
            self.finish(False)
        return response

    def record(self, event, **data):
        item = {'time_utc': datetime.now(timezone.utc).isoformat(), 'event': event,
                'state': self.state, **data}
        self.get_logger().info(json.dumps(item, ensure_ascii=False))
        if self.log_file:
            self.log_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.log_file.flush()

    def transition(self, state, **data):
        self.state = state
        self.state_since = time.monotonic()
        self.record('state', **data)

    def speed(self, linear=0.0, angular=0.0):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(linear), float(angular)
        self.cmd.publish(msg)

    def stop(self):
        self.speed()

    def set_joints(self, arm=None, closed=None):
        # The upstream EP gripper linkage is fixed in this Gazebo model;
        # the vacuum plugin performs the simulated grasp/release.
        if arm is None:
            return
        msg = JointTrajectory()
        msg.header.frame_id = 'world'
        msg.joint_names = ['arm_1_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(arm)]
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0
        msg.points.append(point)
        self.joints.publish(msg)

    def switch_grip(self, enabled):
        client = self.grip_clients[self.pick_object]
        if not client.service_is_ready():
            self.fail('Gazebo grip service unavailable')
            return False
        request = SetBool.Request()
        request.data = enabled
        self.record('grip_request', object_name=self.pick_object, enabled=enabled)
        self.grip_future = client.call_async(request)
        return True

    def selected_detection(self):
        if time.monotonic() - self.last_detection_at > 2.0:
            return None
        expected_class = PICK_OBJECTS[self.pick_heading][1]
        choices = [d for d in self.detections if d['class_id'] == expected_class
                   and (self.target_class is None or d['class_id'] == self.target_class)]
        if not choices:
            return None
        return min(choices, key=lambda d: abs(
            (d['xyxy'][0] + d['xyxy'][2]) / (2 * self.image_width) - 0.5))

    def turn_to(self, target):
        error = wrap_radians(target - self.pose[2])
        angular = limited_turn(error)
        self.speed(angular=angular)
        return angular == 0.0

    def elapsed(self):
        return time.monotonic() - self.state_since

    def start_run(self):
        if (self.pose is None or time.monotonic() - self.last_image_at > 3.0
                or not all(client.service_is_ready() for client in self.grip_clients.values())):
            return
        root = Path(self.get_parameter('log_root').value).expanduser()
        self.run_dir = root / ('gazebo-sort-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.run_dir / 'events.jsonl').open('w')
        self.home = self.pose[:2]
        self.home_yaw = self.pose[2]
        self.set_joints(arm=0.0, closed=False)
        self.record('started', log_directory=str(self.run_dir), pick_headings=list(PICK_HEADINGS))
        self.transition('select')

    def fail(self, reason):
        self.stop()
        self.record('error', reason=reason)
        self.transition('stopped', reason=reason)
        self.finish(False)

    def finish(self, completed):
        self.stop()
        if completed:
            self.set_joints(arm=0.0, closed=False)
        result = {'completed': completed, 'all_six_placed': sum(self.counts.values()) == 6,
                  'counts': self.counts, 'unprocessed_headings': self.pending}
        self.record('result', **result)
        if self.run_dir:
            (self.run_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def tick(self):
        if self.state in ('idle', 'done', 'stopped'):
            if self.state == 'idle' and self.requested:
                self.start_run()
            return
        if self.pose is None:
            self.fail('odometry lost')
            return
        if (self.state in ('seek', 'align') and self.heading_seen
                and time.monotonic() - self.heading_search_started > 180):
            self.fail('seen object could not be aligned: ' + self.pick_object)
            return
        if self.elapsed() > 300 and self.state not in ('seek', 'align'):
            self.fail('stage timeout: ' + self.state)
            return
        if self.state in ('return_center', 'turn_place', 'lower') and (
                time.monotonic() - self.last_grasp_at > 2.0 or not self.grasping):
            self.fail('object lost before placement')
            return

        if self.state == 'select':
            if not self.pending:
                self.transition('done')
                self.finish(True)
                return
            current = math.degrees(wrap_radians(self.pose[2] - self.home_yaw))
            self.pick_heading = nearest_heading(self.pending, current)
            self.pick_object = PICK_OBJECTS[self.pick_heading][0]
            self.grasping = False
            self.last_grasp_at = 0.0
            self.scan_index = 0
            self.heading_seen = False
            self.heading_search_started = time.monotonic()
            self.target_class = None
            self.target_yaw = self.home_yaw + math.radians(self.pick_heading)
            self.set_joints(arm=-1.3, closed=False)
            self.record('selected_direction', heading=self.pick_heading,
                        object_name=self.pick_object)
            self.transition('turn_pick')

        elif self.state == 'turn_pick':
            if self.elapsed() < 0.5:
                self.stop()
                return
            if self.turn_to(self.target_yaw):
                self.stop()
                self.set_joints(arm=0.0, closed=False)
                self.transition('seek')

        elif self.state == 'seek':
            if self.elapsed() < 0.8:
                return
            target = self.selected_detection()
            if target:
                self.heading_seen = True
                self.scan_index = 0
                self.target_class = target['class_id']
                self.centered_frames = 0
                self.record('target_seen', class_id=self.target_class, box=target['xyxy'])
                self.transition('align')
            elif self.elapsed() > 4.0:
                self.scan_index += 1
                offsets = (0, 15, -15, 30, -30)
                if self.scan_index >= len(offsets):
                    if not self.heading_seen:
                        self.record('direction_empty', heading=self.pick_heading)
                        self.pending.remove(self.pick_heading)
                        self.transition('select')
                        return
                    self.scan_index = 0
                offset = offsets[self.scan_index]
                self.target_yaw = self.home_yaw + math.radians(self.pick_heading + offset)
                self.set_joints(arm=-1.3, closed=False)
                self.transition('turn_pick')

        elif self.state == 'align':
            target = self.selected_detection()
            if target is None:
                self.stop()
                if self.elapsed() > 10:
                    self.target_class = None
                    self.transition('seek')
                return
            center = (target['xyxy'][0] + target['xyxy'][2]) / (2 * self.image_width)
            error = center - 0.5
            if abs(error) <= 0.04:
                self.stop()
                self.centered_frames += 1
                if self.centered_frames >= 3:
                    self.travel_start = self.pose[:2]
                    self.close_samples = 0
                    self.transition('approach', class_id=self.target_class)
            else:
                self.centered_frames = 0
                self.speed(angular=-math.copysign(min(0.3, max(0.12, abs(error) * 0.8)), error))
            if self.elapsed() > 25:
                self.fail('alignment timeout')

        elif self.state == 'approach':
            if time.monotonic() - self.last_range_at > 2:
                self.fail('infrared range missing')
                return
            traveled = math.dist(self.pose[:2], self.travel_start)
            if traveled > 0.42 or self.elapsed() > 20:
                self.fail('object not reached within 0.42 m')
                return
            if math.isfinite(self.range_m) and self.range_m <= float(self.get_parameter('grasp_range_m').value):
                self.stop()
                self.close_samples += 1
                if self.close_samples >= 3:
                    self.set_joints(closed=True)
                    if self.switch_grip(True):
                        self.transition('grasp_wait', distance_m=self.range_m)
            else:
                self.close_samples = 0
                self.speed(linear=0.06)

        elif self.state == 'grasp_wait':
            self.stop()
            if self.grip_future and self.grip_future.done():
                try:
                    response = self.grip_future.result()
                except Exception as exc:
                    self.fail('grip service failed: ' + str(exc))
                    return
                if not response.success:
                    self.fail('Gazebo grip did not switch on')
                    return
                self.grip_future = None
            if self.grip_future is None and self.elapsed() >= 0.8 and self.grasping:
                self.set_joints(arm=-1.3, closed=True)
                self.transition('lift')
            elif self.elapsed() > 4:
                self.fail('object was not grasped')

        elif self.state == 'lift':
            if self.elapsed() >= 2.2:
                if not self.grasping:
                    self.fail('object lost during lift')
                else:
                    self.transition('return_center')

        elif self.state == 'return_center':
            distance = math.dist(self.pose[:2], self.home)
            if distance <= 0.03:
                self.stop()
                klass = 'ball' if self.target_class == 32 else 'bottle'
                slot = self.counts[klass]
                if slot >= 3:
                    self.fail('no free placement slot for ' + klass)
                    return
                self.target_yaw = self.home_yaw + math.radians(PLACE_HEADINGS[klass][slot])
                self.transition('turn_place', class_name=klass, slot=slot + 1)
            else:
                self.speed(linear=-0.055)

        elif self.state == 'turn_place':
            if self.turn_to(self.target_yaw):
                self.stop()
                self.set_joints(arm=0.0, closed=True)
                self.transition('lower')

        elif self.state == 'lower':
            if self.elapsed() >= 1.0:
                if self.switch_grip(False):
                    self.set_joints(closed=False)
                    self.transition('release')

        elif self.state == 'release':
            if self.grip_future and self.grip_future.done():
                try:
                    response = self.grip_future.result()
                except Exception as exc:
                    self.fail('release service failed: ' + str(exc))
                    return
                if not response.success:
                    self.fail('Gazebo grip did not switch off')
                    return
                self.grip_future = None
            if self.grip_future is None and self.elapsed() >= 1.0:
                self.set_joints(arm=-1.3, closed=False)
                self.transition('retract')

        elif self.state == 'retract':
            if self.elapsed() >= 1.0:
                klass = 'ball' if self.target_class == 32 else 'bottle'
                self.counts[klass] += 1
                self.pending.remove(self.pick_heading)
                self.record('placed', class_name=klass, heading=self.pick_heading,
                            slot=self.counts[klass], counts=dict(self.counts))
                self.transition('select')


def main(args=None):
    rclpy.init(args=args)
    node = SimSorter()
    try:
        rclpy.spin(node)
    finally:
        node.stop()
        if node.log_file:
            node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()
