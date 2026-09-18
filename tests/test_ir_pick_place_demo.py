import importlib.util
import builtins
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "ir_pick_place_demo", ROOT / "scripts" / "ir_pick_place_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestConfiguration(unittest.TestCase):
    def test_default_configuration_and_raw_conversion(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        self.assertEqual(cfg["infrared"]["threshold_mm"], 20)
        self.assertEqual(cfg["arm"]["distal_hold_raw"], 1073)
        self.assertEqual((cfg["arm"]["distal_servo_id"], cfg["arm"]["distal_feedback_slot"]), (2, 1))
        self.assertEqual((cfg["arm"]["base_servo_id"], cfg["arm"]["base_feedback_slot"]), (1, 0))
        self.assertIsNone(cfg["arm"]["base_extended_raw"])
        self.assertIsNone(cfg["arm"]["base_retracted_raw"])
        self.assertFalse(cfg["arm"]["calibration_verified"])
        self.assertEqual(MODULE.raw_to_sdk_degrees(601), -74)
        self.assertEqual(MODULE.sdk_degrees_to_raw(-74), 603)
        self.assertEqual(MODULE.sdk_degrees_to_wire(-74), 1060)
        self.assertEqual(MODULE.raw_to_sdk_degrees(1073), 9)
        self.assertEqual(MODULE.sdk_degrees_to_raw(9), 1075)
        # Real EP read: DDS 564 paired with get_angle() 99.1 degrees.
        self.assertAlmostEqual(564 * 180 / 1024, 99.1, delta=0.1)
        self.assertEqual(MODULE.raw_to_sdk_degrees(564), -81)

    def test_rejects_invalid_ir_threshold(self):
        cfg = json.loads((ROOT / "config" / "ir_pick_place.json").read_text())
        cfg["infrared"]["threshold_mm"] = 0
        with self.assertRaises(ValueError):
            MODULE.validate_config(cfg)

    def test_execute_with_unverified_config_reaches_sdk_without_motion(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        # Synthetic targets for mocked commands; these are not hardware poses.
        cfg["arm"].update(base_extended_raw=1000, base_retracted_raw=1100)
        real_import = builtins.__import__

        def without_sdk(name, *args, **kwargs):
            if name == "robomaster":
                raise ImportError("test SDK unavailable")
            return real_import(name, *args, **kwargs)

        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["demo", "--execute"]), \
                patch.object(MODULE, "load_config", return_value=cfg), \
                patch("builtins.__import__", side_effect=without_sdk), \
                patch("sys.stdout", output), patch("sys.stderr", errors):
            with self.assertRaises(SystemExit) as stopped:
                MODULE.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("本次按配置执行", output.getvalue())
        self.assertIn("RoboMaster SDK unavailable", errors.getvalue())

    def test_missing_pose_targets_stop_before_sdk_import(self):
        with patch("sys.argv", ["demo", "--execute"]), \
                patch("sys.stderr", new_callable=io.StringIO) as errors:
            with self.assertRaises(SystemExit) as stopped:
                MODULE.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("旧目标已作废", errors.getvalue())

    def test_direct_cycle_cannot_start_with_missing_targets(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        demo = MODULE.Demo(None, cfg, None, lambda *args, **kwargs: None)
        with patch.object(demo, "lock_initial_distal") as lock:
            with self.assertRaisesRegex(ValueError, "外伸最低"):
                demo.run()
            lock.assert_not_called()


class FakeAction:
    has_succeeded = True

    def wait_for_completed(self, timeout):
        return True


class FakeServo:
    def __init__(self):
        self.commands = []

    def moveto(self, index, angle):
        self.commands.append((index, angle))
        return FakeAction()


class FakeFeedback:
    def __init__(self, cfg):
        self.cfg = cfg

    def servo_raw(self, slot, freshness=1.0):
        arm = self.cfg["arm"]
        return arm["distal_hold_raw"] if slot == arm["distal_feedback_slot"] else self.target


class TestServoGuard(unittest.TestCase):
    def test_failed_action_is_reported_even_if_feedback_matches_target(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        action = FakeAction()
        action.has_succeeded = False
        action.state = "action_failed"
        action.failure_reason = None
        servo = FakeServo()
        servo.moveto = lambda **kwargs: action
        bot = type("Bot", (), {"servo": servo})()
        feedback = FakeFeedback(cfg)
        feedback.target = cfg["arm"]["distal_hold_raw"]
        events = []
        demo = MODULE.Demo(bot, cfg, feedback, lambda stage, **values: events.append((stage, values)))
        with self.assertRaisesRegex(RuntimeError, "state=action_failed"):
            demo.lock_initial_distal()
        self.assertFalse(demo.distal_locked)
        self.assertEqual(events[-1][0], "lock_distal_action_failed")
        self.assertEqual(events[-1][1]["actual_raw"], 1073)

    def test_distal_is_commanded_once_then_only_base_moves(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["arm"]["base_retracted_raw"] = 1100  # Fake hardware only.
        bot = type("Bot", (), {"servo": FakeServo()})()
        feedback = FakeFeedback(cfg)
        feedback.target = cfg["arm"]["distal_hold_raw"]
        demo = MODULE.Demo(bot, cfg, feedback, lambda *args, **kwargs: None)
        demo.lock_initial_distal()
        feedback.target = cfg["arm"]["base_retracted_raw"]
        demo.move_base(cfg["arm"]["base_retracted_raw"], "test")
        ids = [item[0] for item in bot.servo.commands]
        self.assertEqual(ids, [cfg["arm"]["distal_servo_id"], cfg["arm"]["base_servo_id"]])

    def test_direct_distal_command_is_rejected_after_lock(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        bot = type("Bot", (), {"servo": FakeServo()})()
        demo = MODULE.Demo(bot, cfg, FakeFeedback(cfg), lambda *args, **kwargs: None)
        demo.distal_locked = True
        with self.assertRaises(RuntimeError):
            demo.move_servo(2, 1, 1073, "forbidden")
        self.assertEqual(bot.servo.commands, [])


class TestInfraredSamples(unittest.TestCase):
    def test_three_fresh_20mm_samples_trigger_but_21mm_does_not(self):
        for distance, should_trigger in [(20, True), (21, False)]:
            with self.subTest(distance=distance):
                cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
                cfg["infrared"]["wait_timeout_s"] = 0.5
                clock = [10.0]
                feedback = type("Feedback", (), {
                    "distance": [distance, 0, 0, 0], "distance_time": 10.0,
                })()

                class Condition:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def wait(self, timeout):
                        clock[0] += timeout
                        feedback.distance_time = clock[0]

                feedback.condition = Condition()
                events = []
                demo = MODULE.Demo(None, cfg, feedback, lambda stage, **kw: events.append(stage))
                with patch.object(MODULE.time, "monotonic", side_effect=lambda: clock[0]):
                    if should_trigger:
                        demo.wait_for_object()
                    else:
                        with self.assertRaises(RuntimeError):
                            demo.wait_for_object()
                self.assertEqual("grasp_distance_confirmed" in events, should_trigger)

    def test_one_sample_cannot_be_counted_three_times(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["infrared"]["wait_timeout_s"] = 0.5
        cfg["infrared"]["freshness_timeout_s"] = 0.1
        clock = [10.0]

        class Condition:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def wait(self, timeout):
                clock[0] += timeout

        feedback = type("Feedback", (), {
            "condition": Condition(), "distance": [20, 0, 0, 0], "distance_time": 10.0,
        })()
        events = []
        demo = MODULE.Demo(None, cfg, feedback, lambda stage, **kw: events.append(stage))
        with patch.object(MODULE.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(RuntimeError):
                demo.wait_for_object()
        self.assertNotIn("grasp_distance_confirmed", events)


class TestSequence(unittest.TestCase):
    def test_cycle_command_order_with_synthetic_targets(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["arm"].update(base_extended_raw=1000, base_retracted_raw=1100)

        class SequenceDemo(MODULE.Demo):
            def __init__(self):
                self.cfg = cfg
                self.arm = cfg["arm"]
                self.events = []

            def lock_initial_distal(self):
                self.events.append(("distal_lock", 1073))

            def initial_state(self, prefix):
                self.events.append((prefix, "extended_open"))

            def wait_for_object(self):
                self.events.append(("infrared", 20))

            def gripper(self, opened, stage):
                self.events.append((stage, "open" if opened else "closed"))

            def move_base(self, target_raw, stage):
                self.events.append((stage, target_raw))

            def turn(self, degrees, stage):
                self.events.append((stage, degrees))

            def record(self, stage, **values):
                pass

        demo = SequenceDemo()
        demo.run()
        self.assertEqual(demo.events, [
            ("distal_lock", 1073),
            ("start", "extended_open"),
            ("infrared", 20),
            ("grasp_close", "closed"),
            ("carry_retract", 1100),
            ("turn_to_place", -90.0),
            ("place_extend", 1000),
            ("place_release", "open"),
            ("return_retract", 1100),
            ("turn_to_start", 90.0),
            ("finish", "extended_open"),
        ])


if __name__ == "__main__":
    unittest.main()
