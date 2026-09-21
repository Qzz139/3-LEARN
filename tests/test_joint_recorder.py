# 纯软件回归测试：使用模拟模型、时钟及SDK接口，不发送实机动作。
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


# 验证只读记录的深拷贝、新鲜度和部分结果保存。
class RecorderTests(unittest.TestCase):
    # 改变SDK复用数组后，已保存的反馈仍应保持原值。
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

    # 离线、过期及格式错误反馈均不能被报告为完整记录。
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

    # 用仅有只读接口的模拟机器人验证订阅失败仍能保存部分记录。
    def test_record_uses_only_read_interfaces_and_saves_partial(self):
        # 仅提供舵机只读订阅接口，避免测试依赖实机控制。
        class Servo:
            # 模拟订阅成功并推送测试舵机反馈。
            def sub_servo_info(self, freq, callback):
                for _ in range(2):
                    callback(([1, 1, 0, 0], [0]*4, [123, 456, 0, 0]))
                return True

            # 模拟取消舵机订阅，不访问SDK网络。
            def unsub_servo_info(self):
                pass

        # 模拟位置订阅失败，检验记录程序的异常收尾。
        class Arm:
            # 模拟机械臂位置订阅被拒绝，用于验证部分记录仍被保存。
            def sub_position(self, freq, callback):
                return False

            # 模拟位置订阅清理，不访问SDK网络。
            def unsub_position(self):
                pass

        # 提供只读记录所需的模拟机器人接口。
        class Bot:
            servo, robotic_arm = Servo(), Arm()

            # 模拟连接成功，测试无需EP在线。
            def initialize(self, conn_type):
                return True

            # 模拟关闭机器人连接，不触及真实设备。
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
