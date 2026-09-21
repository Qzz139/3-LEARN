"""Check detection fields for both Foxy-era and newer vision_msgs layouts."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ros2_ws/src/ep_sorting'))
from ep_sorting.vision_codec import from_detection_array, to_detection_array  # noqa: E402


class Array:
    def __init__(self):
        self.detections = []


class Detection:
    def __init__(self):
        self.bbox = SimpleNamespace(center=SimpleNamespace(x=0.0, y=0.0),
                                    size_x=0.0, size_y=0.0)
        self.results = []


class OldHypothesis:
    def __init__(self):
        self.id = ''
        self.score = 0.0


class NewHypothesis:
    def __init__(self):
        self.hypothesis = SimpleNamespace(class_id='', score=0.0)


class CodecTests(unittest.TestCase):
    def test_round_trip_both_message_layouts(self):
        predictions = [{'class_id': 32, 'confidence': .82, 'xyxy': [20, 30, 60, 70]}]
        for kind in (OldHypothesis, NewHypothesis):
            message = to_detection_array(object(), predictions, Array, Detection, kind)
            result = from_detection_array(message, 640, 480)
            self.assertEqual(result[0]['class_id'], 32)
            self.assertEqual(result[0]['xyxy'], [20.0, 30.0, 60.0, 70.0])
            self.assertEqual(result[0]['image_width'], 640)


if __name__ == '__main__':
    unittest.main()
