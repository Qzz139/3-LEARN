"""Subscribe to EP images, run yolo26m, and publish Detection2DArray."""
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .vision_codec import to_detection_array


class DetectorNode(Node):
    def __init__(self):
        super().__init__('ep_detector')
        from ultralytics import YOLO
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence', 0.3)
        self.declare_parameter('image_size', 640)
        model_path = Path(self.get_parameter('model_path').value)
        if not model_path.is_file():
            raise FileNotFoundError('YOLO model missing: ' + str(model_path))
        self.model = YOLO(str(model_path))
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                           imgsz=int(self.get_parameter('image_size').value), verbose=False)
        self.publisher = self.create_publisher(Detection2DArray, '/ep/detections', 10)
        image_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(Image, '/ep/camera/image_raw',
                                                     self.on_image, image_qos)
        self.get_logger().info('YOLO ready: ' + str(model_path))

    def on_image(self, message):
        if message.encoding != 'bgr8' or message.step < message.width * 3:
            self.get_logger().error('expected bgr8 camera image')
            return
        try:
            frame = np.frombuffer(message.data, dtype=np.uint8).reshape(
                message.height, message.step)[:, :message.width * 3].reshape(
                    message.height, message.width, 3).copy()
            predictions = []
            for result in self.model.predict(frame,
                                             conf=float(self.get_parameter('confidence').value),
                                             imgsz=int(self.get_parameter('image_size').value),
                                             verbose=False):
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    predictions.append({'class_id': int(box.cls[0]),
                                        'confidence': float(box.conf[0]),
                                        'xyxy': [float(v) for v in box.xyxy[0].tolist()]})
            output = to_detection_array(message.header, predictions, Detection2DArray,
                                        Detection2D, ObjectHypothesisWithPose)
            self.publisher.publish(output)
        except Exception as exc:
            self.get_logger().error('YOLO inference failed: {}'.format(exc))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DetectorNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
