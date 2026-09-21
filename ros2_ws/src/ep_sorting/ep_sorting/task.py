"""Small ROS 2 task client; launching alone never moves the robot."""
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

from ep_sorting_interfaces.action import SortObjects


class TaskNode(Node):
    def __init__(self):
        super().__init__('ep_sorting_task')
        self.declare_parameter('execute', False)
        self.client = ActionClient(self, SortObjects, '/ep/sort_objects')
        self.sent = False
        self.detector_ready = False
        self.create_subscription(Detection2DArray, '/ep/detections', self.on_detections, 10)
        if self.get_parameter('execute').value:
            self.timer = self.create_timer(.5, self.maybe_start)
        else:
            self.get_logger().info('Ready; send an explicit /ep/sort_objects action goal to run.')

    def maybe_start(self):
        if self.sent or not self.detector_ready or not self.client.wait_for_server(timeout_sec=.1):
            return
        self.sent = True
        self.timer.cancel()
        goal = SortObjects.Goal()
        goal.execute = True
        self.client.send_goal_async(goal, feedback_callback=self.on_feedback).add_done_callback(self.on_goal)

    def on_detections(self, _message):
        self.detector_ready = True

    def on_goal(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Sorting goal rejected')
            return
        handle.get_result_async().add_done_callback(self.on_result)

    def on_feedback(self, message):
        item = message.feedback
        self.get_logger().info('{}: {} (placed {})'.format(item.stage, item.state, item.placed))

    def on_result(self, future):
        result = future.result().result
        self.get_logger().info('completed={} balls={} bottles={} log={} error={}'.format(
            result.completed, result.balls_placed, result.bottles_placed,
            result.log_directory, result.error))


def main(args=None):
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
