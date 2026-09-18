import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "recorder", Path(__file__).resolve().parents[1] / "scripts/record_joint_state.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecorderTests(unittest.TestCase):
    def test_copies_sdk_lists_and_keeps_raw_values(self):
        recorder = module.Recorder(io.StringIO())
        data = ([1, 1, 0, 0], [0, 0, 0, 0], [1024, 700, 0, 0])
        for _ in range(2):
            recorder.receive("servo", data)
            recorder.receive("arm", (123, 4294967295))
        data[2][0] = 99
        snapshot = recorder.snapshot()
        self.assertTrue(snapshot["feedback_complete"])
        self.assertEqual(snapshot["latest"]["servo"]["data"]["angle_raw"][0], 1024)
        self.assertEqual(snapshot["latest"]["arm"]["data"]["y_raw_sdk_mm"], 4294967295)

    def test_invalid_or_stale_data_is_not_complete(self):
        recorder = module.Recorder(io.StringIO())
        for _ in range(2):
            recorder.receive("servo", ([0, 0, 0, 0], [0]*4, [0]*4))
            recorder.receive("arm", (0, 0))
        self.assertFalse(recorder.snapshot()["feedback_complete"])
        recorder.receive("servo", ([1, 1, 0, 0], [0]*4, [100]*4))
        recorder.received["servo"] -= 2
        self.assertFalse(recorder.snapshot()["feedback_complete"])
        recorder.receive("servo", ([1], [0], [0]))
        self.assertTrue(recorder.snapshot()["errors"])

    def test_record_uses_only_read_interfaces_and_saves_partial(self):
        class Servo:
            def sub_servo_info(self, freq, callback):
                for _ in range(2):
                    callback(([1, 1, 0, 0], [0]*4, [123, 456, 0, 0]))
                return True

            def unsub_servo_info(self):
                pass

        class Arm:
            def sub_position(self, freq, callback):
                return False

            def unsub_position(self):
                pass

        class Bot:
            servo, robotic_arm = Servo(), Arm()

            def initialize(self, conn_type):
                return True

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "record"
            code = module.record(Bot(), destination, 0.001, 10, "test", "ap", "fake")
            data = json.loads((destination / "snapshot.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(data["status"], "partial")
            self.assertFalse(data["motion_commands_sent"])
            self.assertEqual(data["counts"]["servo"], 2)
            with self.assertRaises(FileExistsError):
                module.record(Bot(), destination, 0.001, 10, "test", "ap", "fake")


if __name__ == "__main__":
    unittest.main()
