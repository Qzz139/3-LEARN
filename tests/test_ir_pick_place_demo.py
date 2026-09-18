# 纯软件回归测试：使用模拟模型、时钟及SDK接口，不发送实机动作。
import importlib.util
import builtins
import io
import json
import queue
import tempfile
import threading
import types
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "ir_pick_place_demo", ROOT / "scripts" / "ir_pick_place_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


# 验证配置和入口执行条件，测试不会连接真实机器人。
class TestConfiguration(unittest.TestCase):
    # 核对实机舵机映射与姿态参数，并区分DDS、SDK角度及协议编码。
    def test_default_configuration_and_raw_conversion(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        self.assertEqual(cfg["infrared"]["threshold_mm"], 20)
        self.assertEqual(cfg["arm"]["distal_hold_raw"], 1073)
        self.assertEqual((cfg["arm"]["distal_servo_id"], cfg["arm"]["distal_feedback_slot"]), (2, 1))
        self.assertEqual((cfg["arm"]["base_servo_id"], cfg["arm"]["base_feedback_slot"]), (1, 0))
        self.assertEqual(cfg["arm"]["base_extended_raw"], 600)
        self.assertEqual(cfg["arm"]["base_retracted_raw"], 1190)
        self.assertTrue(cfg["arm"]["calibration_verified"])
        self.assertEqual(MODULE.raw_to_sdk_degrees(601), -74)
        self.assertEqual(MODULE.sdk_degrees_to_raw(-74), 603)
        self.assertEqual(MODULE.sdk_degrees_to_wire(-74), 1060)
        self.assertEqual(MODULE.raw_to_sdk_degrees(1073), 9)
        self.assertEqual(MODULE.sdk_degrees_to_raw(9), 1075)
        # Real EP read: DDS 564 paired with get_angle() 99.1 degrees.
        self.assertAlmostEqual(564 * 180 / 1024, 99.1, delta=0.1)
        self.assertEqual(MODULE.raw_to_sdk_degrees(564), -81)

    # 无效红外阈值必须在连接机器人前被拒绝。
    def test_rejects_invalid_ir_threshold(self):
        cfg = json.loads((ROOT / "config" / "ir_pick_place.json").read_text())
        cfg["infrared"]["threshold_mm"] = 0
        with self.assertRaises(ValueError):
            MODULE.validate_config(cfg)

    # 未标定提示不再拦截执行，模拟缺少SDK确保测试不会发送实机动作。
    def test_execute_with_unverified_config_reaches_sdk_without_motion(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["arm"]["calibration_verified"] = False
        # Synthetic targets for mocked commands; these are not hardware poses.
        cfg["arm"].update(base_extended_raw=1000, base_retracted_raw=1100)
        real_import = builtins.__import__

        # 仅拦截RoboMaster导入，模拟缺少SDK而不执行机器人连接。
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

    # 缺失姿态目标必须在SDK导入和机器人连接之前终止。
    def test_missing_pose_targets_stop_before_sdk_import(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["arm"]["base_retracted_raw"] = None
        with patch("sys.argv", ["demo", "--execute"]), \
                patch.object(MODULE, "load_config", return_value=cfg), \
                patch("sys.stderr", new_callable=io.StringIO) as errors:
            with self.assertRaises(SystemExit) as stopped:
                MODULE.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("旧目标已作废", errors.getvalue())

    # 直接调用抓放流程也不能绕过姿态目标检查。
    def test_direct_cycle_cannot_start_with_missing_targets(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["arm"]["base_retracted_raw"] = None
        demo = MODULE.Demo(None, cfg, None, lambda *args, **kwargs: None)
        with patch.object(demo, "lock_initial_distal") as lock:
            with self.assertRaisesRegex(ValueError, "外伸最低"):
                demo.run()
            lock.assert_not_called()


# 提供SDK动作的模拟完成状态，测试可覆写成功标志。
class FakeAction:
    has_succeeded = True

    # 返回预设的动作完成状态，不等待实际舵机。
    def wait_for_completed(self, timeout):
        return True


# 记录发送给各舵机的模拟命令。
class FakeServo:
    # 初始化该测试替身所需的参数、缓存或命令记录。
    def __init__(self):
        self.commands = []

    # 记录模拟定位命令并返回假动作，便于核对舵机ID和角度。
    def moveto(self, index, angle):
        self.commands.append((index, angle))
        return FakeAction()


# 为固定关节和目标关节提供可控的原始反馈。
class FakeFeedback:
    # 初始化该测试替身所需的参数、缓存或命令记录。
    def __init__(self, cfg):
        self.cfg = cfg

    # 按测试槽位返回固定关节或当前目标的模拟原始角度。
    def servo_raw(self, slot, freshness=1.0):
        arm = self.cfg["arm"]
        return arm["distal_hold_raw"] if slot == arm["distal_feedback_slot"] else self.target


# 验证抓夹侧只定位一次及舵机动作失败处理。
class TestServoGuard(unittest.TestCase):
    # 即使反馈角度吻合，也不能把SDK失败的定位动作当作成功。
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

    # 确认抓夹侧只定位一次，后续定位命令只发给底盘侧。
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

    # 锁定后直接向抓夹侧定位必须被拦截，且不产生模拟命令。
    def test_direct_distal_command_is_rejected_after_lock(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        bot = type("Bot", (), {"servo": FakeServo()})()
        demo = MODULE.Demo(bot, cfg, FakeFeedback(cfg), lambda *args, **kwargs: None)
        demo.distal_locked = True
        with self.assertRaises(RuntimeError):
            demo.move_servo(2, 1, 1073, "forbidden")
        self.assertEqual(bot.servo.commands, [])


# 用模拟时钟检查连续新测距样本的判定。
class TestInfraredSamples(unittest.TestCase):
    # 验证旧demo的20mm原始阈值及三个新样本条件，不等同实际表面距离。
    def test_three_fresh_20mm_samples_trigger_but_21mm_does_not(self):
        for distance, should_trigger in [(20, True), (21, False)]:
            with self.subTest(distance=distance):
                cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
                cfg["infrared"]["wait_timeout_s"] = 0.5
                clock = [10.0]
                feedback = type("Feedback", (), {
                    "distance": [distance, 0, 0, 0], "distance_time": 10.0,
                })()

                # 模拟条件变量等待，控制测试时间和反馈更新时间。
                class Condition:
                    # 模拟条件变量进入上下文，不获取真实硬件资源。
                    def __enter__(self):
                        return self

                    # 模拟上下文退出，不干预测试中的异常传递。
                    def __exit__(self, *args):
                        pass

                    # 推进模拟时钟，并按当前场景模拟反馈是否更新。
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

    # 冻结反馈时间戳，确认同一近距样本不能重复凑足三次。
    def test_one_sample_cannot_be_counted_three_times(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["infrared"]["wait_timeout_s"] = 0.5
        cfg["infrared"]["freshness_timeout_s"] = 0.1
        clock = [10.0]

        # 模拟条件变量等待，控制测试时间和反馈更新时间。
        class Condition:
            # 模拟条件变量进入上下文，不获取真实硬件资源。
            def __enter__(self):
                return self

            # 模拟上下文退出，不干预测试中的异常传递。
            def __exit__(self, *args):
                pass

            # 推进模拟时钟，并按当前场景模拟反馈是否更新。
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


# 通过记录模拟事件核对单次抓放动作顺序。
class TestSequence(unittest.TestCase):
    # 用纯模拟姿态核对整轮动作顺序，测试角度不能拿来驱动实机。
    def test_cycle_command_order_with_synthetic_targets(self):
        cfg = MODULE.load_config(ROOT / "config" / "ir_pick_place.json")
        cfg["arm"].update(base_extended_raw=1000, base_retracted_raw=1100)

        # 替换真实硬件操作，只保留抓放流程的事件序列。
        class SequenceDemo(MODULE.Demo):
            # 初始化该测试替身所需的参数、缓存或命令记录。
            def __init__(self):
                self.cfg = cfg
                self.arm = cfg["arm"]
                self.events = []

            # 记录锁定远端关节的模拟动作，不发送实机定位。
            def lock_initial_distal(self):
                self.events.append(("distal_lock", 1073))

            # 记录模拟初始姿态或按场景改变位置反馈，不控制实机。
            def initial_state(self, prefix):
                self.events.append((prefix, "extended_open"))

            # 模拟红外到达抓取条件，跳过真实传感器等待。
            def wait_for_object(self):
                self.events.append(("infrared", 20))

            # 只记录抓夹开闭事件，不发送抓夹命令。
            def gripper(self, opened, stage):
                self.events.append((stage, "open" if opened else "closed"))

            # 只记录底盘侧舵机的模拟目标，用于核对动作顺序。
            def move_base(self, target_raw, stage):
                self.events.append((stage, target_raw))

            # 只记录模拟转角，用于核对分类放置和返回动作。
            def turn(self, degrees, stage):
                self.events.append((stage, degrees))

            # 忽略或记录模拟日志，不触及真实运行报告。
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


# 验证模型冷加载期间结束流程时，视觉子进程能正确收尾。
class TestVisionShutdown(unittest.TestCase):
    # 抓放在模型加载期间结束时，视觉线程仍应记录已缓存的一帧。
    def test_pending_frame_is_logged_when_cycle_finishes_during_model_load(self):
        frames, stopped, calls = queue.Queue(), threading.Event(), []
        frames.put(object())
        stopped.set()
        # 提供空检测结果的模型替身，记录是否调用推理。
        class Model:
            # 记录模拟模型被调用的次数，返回空检测结果。
            def predict(self, frame, **kwargs):
                calls.append(frame)
                return []
        fake = types.SimpleNamespace(YOLO=lambda name: Model())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detections.jsonl'
            with patch.dict('sys.modules', {'ultralytics': fake}), patch('sys.stdout', new_callable=io.StringIO):
                MODULE.yolo_worker(frames, stopped, 'test.pt', .25, 640, str(path))
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual([event['stage'] for event in events], ['model_ready', 'detections'])

    # 退出信号已到且无缓存图像时，推理子进程应直接结束。
    def test_stopped_worker_without_frames_exits_without_inference(self):
        frames, stopped = queue.Queue(), threading.Event()
        stopped.set()
        fake = types.SimpleNamespace(YOLO=lambda name: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detections.jsonl'
            with patch.dict('sys.modules', {'ultralytics': fake}), patch('sys.stdout', new_callable=io.StringIO):
                MODULE.yolo_worker(frames, stopped, 'test.pt', .25, 640, str(path))
            events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([event['stage'] for event in events], ['model_ready'])


if __name__ == "__main__":
    unittest.main()
