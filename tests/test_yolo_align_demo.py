# 纯软件回归测试：使用模拟模型、时钟及SDK接口，不发送实机动作。
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('align_demo', ROOT / 'scripts/yolo_align_pick_place_demo.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


# 读取仓库配置，确保测试使用当前稳定版参数。
def config():
    return M.validate_alignment(json.loads((ROOT / 'config/yolo_align_pick_place.json').read_text()))


# 生成可控制类别、中心和大小的检测框，不调用真实模型。
def box(cx, cls=32, width=1280, height=720, scale=1):
    return dict(class_id=cls, confidence=.9, xyxy=[cx-60*scale, 300, cx+60*scale, 420],
                image_width=width, image_height=height)


# 用模拟图像与反馈验证视觉对准、接近、停车和路径回退。
class TestAlignment(unittest.TestCase):
    # 模拟连续图传，核对后台持续取帧并在请求时返回新图像。
    def test_camera_feed_drains_while_control_is_idle_and_returns_a_new_frame(self):
        # 持续生成模拟图像，验证新帧采集线程。
        class Camera:
            count = 0
            # 模拟不断产生新帧，检验后台采集没有因控制空闲而停下。
            def read_cv2_image(self, **kwargs):
                M.time.sleep(.005)
                self.count += 1
                return self.count
        camera = Camera()
        feed = M.CameraFeed(camera)
        feed.start()
        try:
            first, first_time = feed.read_latest()
            M.time.sleep(.03)
            second, second_time = feed.read_latest()
            self.assertGreater(second, first + 1)
            self.assertGreater(second_time, first_time)
        finally:
            feed.stop()
        self.assertFalse(feed.thread.is_alive())

    # 验证图像偏差对应SDK转向符号、死区及单步角度限制。
    def test_image_left_turns_left_and_right_turns_right_with_limited_steps(self):
        a = config()['alignment']
        self.assertEqual(M.correction_degrees(M.pixel_error(box(320), .5), a), 5)
        self.assertEqual(M.correction_degrees(M.pixel_error(box(960), .5), a), -5)
        self.assertEqual(M.correction_degrees(M.pixel_error(box(650), .5), a), 0)
        self.assertEqual(M.correction_degrees(-.03, a), 0)
        self.assertEqual(M.correction_degrees(.04, a), 0)
        self.assertEqual(M.correction_degrees(-.04, a), 0)
        self.assertGreater(M.correction_degrees(-.041, a), 0)
        self.assertLess(M.correction_degrees(.041, a), 0)

    # 瓶子模式必须选择并持续跟踪瓶子，同时使用左侧放置方向。
    def test_bottle_mode_selects_and_tracks_bottle_without_grabbing_ball(self):
        cfg = M.validate_alignment(M.select_target_mode(config(), 'bottle'))
        a = cfg['alignment']
        ball, bottle = box(640), box(760, 39)
        self.assertEqual(a['target_class_ids'], [39])
        self.assertEqual(cfg['chassis']['place_turn_degrees'], 90)
        self.assertEqual(cfg['arm'], config()['arm'])
        self.assertEqual(M.choose_target([ball, bottle], a), bottle)
        self.assertIsNone(M.choose_target([ball], a, bottle))
        next_bottle = box(720, 39)
        self.assertEqual(M.choose_target([ball, next_bottle], a, bottle), next_bottle)
        cfg = M.select_target_mode(cfg, 'ball')
        self.assertEqual(cfg['alignment']['target_class_ids'], [32])
        self.assertEqual(cfg['chassis']['place_turn_degrees'], -90)

    def test_locked_bottle_survives_same_box_vase_misclassification(self):
        a = M.validate_alignment(M.select_target_mode(config(), 'bottle'))['alignment']
        bottle = box(480, 39, width=960, height=540)
        vase = box(482, 75, width=960, height=540)
        self.assertIsNone(M.choose_target([vase], a))
        continued = M.choose_target([vase], a, bottle)
        self.assertEqual(continued['class_id'], 39)
        self.assertEqual(continued['observed_class_id'], 75)
        self.assertIsNone(M.choose_target([box(600, 75, width=960, height=540)], a, bottle))
        self.assertIsNone(M.choose_target([box(482, 75, width=960, height=540, scale=2)], a, bottle))
        self.assertEqual(M.choose_target([bottle, vase], a, bottle), bottle)

    # 球模式排除瓶子，并保持原目标连续性而不换到另一只球。
    def test_target_selection_excludes_bottles_and_tracks_one_ball(self):
        a = config()['alignment']
        previous = box(320)
        bottle, other_ball, same_ball = box(640, 39), box(1000), box(380)
        self.assertEqual(M.choose_target([bottle, same_ball, other_ball], a), same_ball)
        self.assertEqual(M.choose_target([other_ball, same_ball], a, previous), same_ball)
        self.assertIsNone(M.choose_target([other_ball], a, previous))
        self.assertIsNone(M.choose_target([box(340, scale=10)], a, previous))

    # 验证跨±180°的角度处理，以及相对启动朝向的放置补偿。
    def test_heading_wrap_and_place_compensation_use_startup_reference(self):
        cfg = config()
        feedback = type('F', (), {'yaw': lambda self: 160.0})()
        demo = M.AlignDemo(None, cfg, feedback, lambda *args, **kwargs: None, None)
        demo.start_yaw = -170
        self.assertEqual(demo.heading_offset(), 30)
        # Aligned 30deg left -> startup-right placement needs -120deg total, not -90.
        self.assertEqual(M.wrap_degrees(-90 - demo.heading_offset()), -120)

    # 核对本台EP实测的yaw符号与SDK正转角相反。
    def test_measured_yaw_decreases_for_positive_sdk_turn_on_this_ep(self):
        cfg = config()
        feedback = type('F', (), {'yaw': lambda _: .87})()
        demo = M.AlignDemo(None, cfg, feedback, lambda *args, **kwargs: None, None)
        demo.start_yaw = 6.04
        self.assertAlmostEqual(demo.heading_offset(), 5.17)

    # 搜索扫描角不能超出配置的最大范围。
    def test_search_offsets_stay_within_configured_range(self):
        self.assertEqual(M.search_offsets(config()['alignment']), [15, -15, 30, -30, 45, -45, 60, -60])

    # 过期识别结果应丢弃，不能触发转向或抓取。
    def test_stale_vision_cannot_cause_rotation_or_grasp(self):
        cfg = config()
        cfg['alignment']['timeout_s'] = .015
        detector = type('Detector', (), {'detect': lambda *_: ([box(320)], 99)})()
        bot = type('Bot', (), {'camera': None})()
        demo = M.AlignDemo(bot, cfg, None, lambda *args, **kwargs: None, detector)
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'turn') as turn:
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                demo.align_and_wait_for_grasp()
            turn.assert_not_called()

    # 锁定目标持续丢失应停止，不能突然改抓其他球。
    def test_target_loss_stops_instead_of_switching_to_another_ball(self):
        cfg = config()
        # 按场景提供预设检测结果，不加载实际模型。
        class Detector:
            frames = iter([[box(640)], [box(1100)], [box(1100)], [box(1100)]])
            # 按预设次序返回图像识别结果，用于核对搜索和动作先后。
            def detect(self, _):
                return next(self.frames), .01
        demo = M.AlignDemo(type('Bot', (), {'camera': None})(), cfg, None, lambda *args, **kwargs: None, Detector())
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'heading_offset', return_value=0), patch.object(M.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'locked target lost'):
                demo.align_and_wait_for_grasp()

    # 图像连续居中只是条件之一，红外未就绪时不能开始抓取。
    def test_three_centered_frames_still_require_infrared(self):
        cfg = config()
        detector = type('Detector', (), {'detect': lambda *_: ([box(640)], .01)})()
        events = []
        demo = M.AlignDemo(type('Bot', (), {'camera': None})(), cfg, None,
                          lambda stage, **kw: events.append(stage), detector)
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'heading_offset', return_value=0), \
                patch.object(demo, 'infrared_state', side_effect=['wait', 'ready']) as infrared, \
                patch.object(M.time, 'sleep'):
            demo.align_and_wait_for_grasp()
        self.assertEqual(infrared.call_count, 2)
        self.assertEqual(events.count('alignment_stable'), 4)
        self.assertEqual(events[-1], 'alignment_and_distance_ready')

    # 每次前进后必须重新识别并累计居中帧，不能沿用旧结果。
    def test_translation_requires_realigning_and_new_centered_frames(self):
        cfg = config()
        frames = iter([[box(640)]] * 3 + [[box(800)]] + [[box(640)]] * 3)
        detector = type('Detector', (), {'detect': lambda *_: (next(frames), .01)})()
        events = []
        demo = M.AlignDemo(type('Bot', (), {'camera': None})(), cfg, None,
                          lambda stage, **kw: events.append(stage), detector)
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'heading_offset', return_value=0), \
                patch.object(demo, 'infrared_state', side_effect=['far', 'ready']), \
                patch.object(demo, 'approach_step') as advance, patch.object(demo, 'align_turn') as turn, \
                patch.object(M.time, 'sleep'):
            demo.align_and_wait_for_grasp()
        advance.assert_called_once()
        turn.assert_called_once()
        self.assertEqual(events.count('alignment_stable'), 6)

    # SDK动作成功但实际yaw没有变化时，对准动作仍应失败。
    def test_successful_action_without_yaw_progress_is_rejected(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        with patch.object(demo, 'heading_offset', side_effect=[0, .02]), \
                patch.object(demo, 'turn'), patch.object(M.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'without actual yaw progress'):
                demo.align_turn(5)

    # 红外近远混合不稳定时等待，不自动前进或夹紧。
    def test_unstable_near_infrared_never_advances(self):
        cfg = config()
        detector = type('Detector', (), {'detect': lambda *_: ([box(640)], .01)})()
        demo = M.AlignDemo(type('Bot', (), {'camera': None})(), cfg, None, lambda *a, **k: None, detector)
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'heading_offset', return_value=0), \
                patch.object(demo, 'infrared_state', side_effect=['wait', 'ready']), \
                patch.object(demo, 'approach_step') as advance, patch.object(M.time, 'sleep'):
            demo.align_and_wait_for_grasp()
        advance.assert_not_called()

    # 红外确认只使用新时间戳，零值及无效数据不能触发抓取。
    def test_infrared_uses_distinct_new_samples_and_rejects_invalid_data(self):
        cfg = config()
        for values, expected in (([15, 15, 15], 'ready'), ([90, 90, 90], 'far'),
                                 ([15, 90, 15, 90], 'wait'), ([0], 'invalid')):
            clock = [10.0]
            feedback = M.Feedback()
            feedback.distance, feedback.distance_time = (90,), 9.99
            sequence = iter(values)
            # 推进模拟时钟，并按当前场景模拟反馈是否更新。
            def wait(timeout):
                clock[0] += timeout
                value = next(sequence, None)
                if value is not None:
                    feedback.distance, feedback.distance_time = (value,), clock[0]
            demo = M.AlignDemo(None, cfg, feedback, lambda *a, **k: None, None)
            with self.subTest(values=values), patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(feedback.condition, 'wait', side_effect=wait), patch.object(demo, 'assert_distal'):
                if expected == 'invalid':
                    with self.assertRaisesRegex(RuntimeError, 'invalid'):
                        demo.infrared_state()
                elif expected == 'wait':
                    # Alternating samples must not authorize forward drive; an
                    # eventually stale stream is also correctly rejected.
                    with self.assertRaisesRegex(RuntimeError, 'stale'):
                        demo.infrared_state()
                else:
                    self.assertEqual(demo.infrared_state(), expected)

    # 过期里程或红外数据不能继续驱动底盘。
    def test_stale_position_and_infrared_are_rejected(self):
        feedback = M.Feedback()
        feedback.position_value, feedback.position_time = (0, 0), 10
        feedback.distance, feedback.distance_time = (100,), 10
        demo = M.AlignDemo(None, config(), feedback, lambda *a, **k: None, None)
        with patch.object(M.time, 'monotonic', return_value=10.3):
            with self.assertRaisesRegex(RuntimeError, 'position.*stale'):
                demo.position()
            with self.assertRaisesRegex(RuntimeError, 'infrared.*stale'):
                demo.infrared_value()

    # 到达累计前进上限后，不能再发送新的前进速度命令。
    def test_maximum_travel_prevents_another_forward_command(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        demo.travel_m = .2
        with patch.object(demo, 'translate') as move:
            with self.assertRaisesRegex(RuntimeError, 'maximum approach'):
                demo.approach_step()
            move.assert_not_called()

    # 模拟里程和时钟，验证低速前进、反馈停车及按原路径倒车。
    def test_odometry_controls_slow_motion_and_retreats_to_waypoint(self):
        cfg, clock, position, velocity, commands = config(), [10.0], [0.0, 0.0], [0.0], []
        # 按测试场景记录或模拟速度命令；返回值不代表真实底盘运动。
        def drive_speed(x, y, z, timeout):
            velocity[0] = x
            commands.append(x)
            return False  # Actual official no-ACK SDK behavior, not a rejection.
        # 推进模拟时间及场景位置，避免运动测试依赖真实时长。
        def sleep(duration):
            clock[0] += duration
            position[0] += velocity[0] * duration
        chassis = type('Chassis', (), {'drive_speed': staticmethod(drive_speed), 'drive_wheels': staticmethod(lambda **kw: drive_speed(0, 0, 0, kw['timeout']) or True)})()
        feedback = type('F', (), {'velocity_snapshot': lambda _, freshness: ((velocity[0], 0, 0), clock[0]),
                                 'esc_snapshot': lambda _, freshness: (((0,)*4, (0,)*4, (clock[0],)*4), clock[0])})()
        demo = M.AlignDemo(type('Bot', (), {'chassis': chassis})(), cfg, feedback, lambda *a, **k: None, None)
        demo.start_position = (0, 0)
        with patch.object(M.time, 'sleep', side_effect=sleep), \
                patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(demo, 'assert_distal'), patch.object(demo, 'position', side_effect=lambda: tuple(position)), \
                patch.object(demo, 'heading_offset', return_value=0), patch.object(demo, 'infrared_value', return_value=90), \
                patch.object(demo, 'turn_to_offset'):
            demo.approach_step()
            self.assertAlmostEqual(demo.travel_m, .01)
            self.assertEqual(len(demo.approach_path), 1)
            demo.return_to_center()
        self.assertLessEqual(abs(position[0]), cfg['approach']['waypoint_tolerance_m'])
        self.assertEqual(commands[-1], 0)
        self.assertEqual(set(commands), {0, .04, -.03})

    # 允许小幅横向偏差下按投影到达回退点，大幅偏离则停车。
    def test_retreat_can_pass_waypoint_with_small_lateral_error(self):
        cfg, clock, position, velocity = config(), [10.0], [.02, 0.0], [0.0]
        # 按测试场景记录或模拟速度命令；返回值不代表真实底盘运动。
        def drive_speed(x, y, z, timeout):
            velocity[0] = x
            return False
        # 模拟有应答的轮速停车并改变测试中的速度状态。
        def drive_wheels(**kw):
            velocity[0] = 0
            return True
        # 推进模拟时间及场景位置，避免运动测试依赖真实时长。
        def sleep(dt):
            position[0] += velocity[0]*dt
            clock[0] += dt
        chassis = type('C', (), {'drive_speed': staticmethod(drive_speed), 'drive_wheels': staticmethod(drive_wheels)})()
        feedback = type('F', (), {
            'velocity_snapshot': lambda _, freshness: ((0, 0, 0), clock[0]),
            'esc_snapshot': lambda _, freshness: (((0,)*4, (0,)*4, (clock[0],)*4), clock[0])})()
        demo = M.AlignDemo(type('Bot', (), {'chassis': chassis})(), cfg, feedback, lambda *a, **k: None, None)
        with patch.object(M.time, 'sleep', side_effect=sleep), \
                patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(demo, 'assert_distal'), patch.object(demo, 'position', side_effect=lambda: tuple(position)), \
                patch.object(demo, 'heading_offset', return_value=0):
            demo.translate(M.position_distance(tuple(position), (0, .004)), 'return_path_step', return_target=(0, .004))
            self.assertLess(position[0], .002)
            self.assertEqual(position[1], 0)
            position[0] = .02
            with self.assertRaisesRegex(RuntimeError, 'return path diverged'):
                demo.translate(M.position_distance(tuple(position), (0, .05)), 'return_path_step', return_target=(0, .05))
        self.assertEqual(velocity[0], 0)

    # 前进途中看到阈值或传感器异常，都必须执行停车收尾。
    def test_threshold_or_sensor_failure_stops_velocity_segment(self):
        for readings, raises in (([10], False), ([90, RuntimeError('sensor failed')], True)):
            cfg, clock, position, velocity, commands = config(), [10.0], [0.0, 0.0], [0.0], []
            # 按测试场景记录或模拟速度命令；返回值不代表真实底盘运动。
            def drive_speed(x, y, z, timeout):
                velocity[0] = x
                commands.append(x)
                return True
            # 推进模拟时间及场景位置，避免运动测试依赖真实时长。
            def sleep(duration):
                clock[0] += duration
                position[0] += velocity[0] * duration
            feedback = type('F', (), {'velocity_snapshot': lambda _, freshness: ((velocity[0], 0, 0), clock[0]),
                                 'esc_snapshot': lambda _, freshness: (((0,)*4, (0,)*4, (clock[0],)*4), clock[0])})()
            demo = M.AlignDemo(type('Bot', (), {'chassis': type('C', (), {'drive_speed': staticmethod(drive_speed), 'drive_wheels': staticmethod(lambda **kw: drive_speed(0, 0, 0, kw['timeout']) or True)})()})(),
                               cfg, feedback, lambda *a, **k: None, None)
            with self.subTest(raises=raises), patch.object(M.time, 'sleep', side_effect=sleep), \
                    patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(demo, 'assert_distal'), patch.object(demo, 'position', side_effect=lambda: tuple(position)), \
                    patch.object(demo, 'heading_offset', return_value=0), patch.object(demo, 'infrared_value', side_effect=readings):
                if raises:
                    with self.assertRaisesRegex(RuntimeError, 'sensor failed'):
                        demo.approach_step()
                else:
                    demo.approach_step()
            self.assertEqual(commands[-1], 0)
            if not raises:
                self.assertEqual(commands, [0])

    # 倒序恢复各路段朝向并回退后，再核对启动中心容差。
    def test_return_replays_headings_in_reverse_before_checking_center(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        demo.start_position = (0, 0)
        demo.approach_path = [dict(start=(0, 0), heading=20), dict(start=(.005, 0), heading=25)]
        position = [( .01, 0)]
        # 记录回退目标及更新模拟位置，不发送底盘动作。
        def retreat(distance, stage, return_target):
            position[0] = return_target
        with patch.object(demo, 'position', side_effect=lambda: position[0]), \
                patch.object(demo, 'translate', side_effect=retreat) as move, \
                patch.object(demo, 'turn_to_offset') as turn:
            demo.return_to_center()
        self.assertEqual([c[0][0] for c in turn.call_args_list], [25, 20])
        self.assertEqual([c[1]['return_target'] for c in move.call_args_list], [(.005, 0), (0, 0)])

    # 里程无进展时终止运动，并发送停车命令。
    def test_stalled_odometry_stops_chassis(self):
        cfg, clock, commands = config(), [10.0], []
        # 按测试场景记录或模拟速度命令；返回值不代表真实底盘运动。
        def drive_speed(x, y, z, timeout):
            commands.append(x)
            return True
        # 推进模拟时间及场景位置，避免运动测试依赖真实时长。
        def sleep(duration):
            clock[0] += duration
        demo = M.AlignDemo(type('Bot', (), {'chassis': type('C', (), {'drive_speed': staticmethod(drive_speed), 'drive_wheels': staticmethod(lambda **kw: drive_speed(0, 0, 0, kw['timeout']) or True)})()})(),
                           cfg, None, lambda *a, **k: None, None)
        with patch.object(M.time, 'sleep', side_effect=sleep), \
                patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(demo, 'assert_distal'), patch.object(demo, 'position', return_value=(0, 0)), \
                patch.object(demo, 'heading_offset', return_value=0), patch.object(demo, 'infrared_value', return_value=90):
            with self.assertRaisesRegex(RuntimeError, 'no odometry progress'):
                demo.approach_step()
        self.assertEqual(commands[-1], 0)

    # 模拟速度停车无效的情形，必须用有应答的四轮零转速停车。
    def test_acknowledged_wheel_stop_brakes_when_velocity_stop_is_ignored(self):
        velocity, requests = [.04], []
        # 按测试场景记录或模拟速度命令；返回值不代表真实底盘运动。
        def drive_speed(**kw):
            return False
        # 模拟有应答的轮速停车并改变测试中的速度状态。
        def drive_wheels(**kw):
            requests.append(kw)
            velocity[0] = 0
            return True
        chassis = type('C', (), {'drive_speed': staticmethod(drive_speed),
                                'drive_wheels': staticmethod(drive_wheels)})()
        demo = M.AlignDemo(type('Bot', (), {'chassis': chassis})(), config(), None, lambda *a, **k: None, None)
        demo.stop_translation()
        self.assertEqual(velocity[0], 0)
        self.assertEqual(requests, [dict(w1=0, w2=0, w3=0, w4=0, timeout=.2)])
        with patch.object(chassis, 'drive_wheels', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'stop command was not acknowledged'):
                demo.stop_translation()

    # 轮速和编码器已稳定时，不应仅因速度估计长尾而拒绝停稳。
    def test_stable_encoders_confirm_stop_despite_velocity_estimate_lag(self):
        cfg, clock, events = config(), [10.0], []
        feedback = type('F', (), {
            'velocity_snapshot': lambda _, freshness: ((-.08, 0, 0), clock[0]),
            'esc_snapshot': lambda _, freshness: (((-3, 1, 0, 2), (32765, 100, 200, 300), (clock[0],)*4), clock[0])})()
        demo = M.AlignDemo(None, cfg, feedback, lambda stage, **kw: events.append((stage, kw)), None)
        with patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(M.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)), \
                patch.object(demo, 'assert_distal'):
            demo.wait_stationary()
        self.assertEqual(events[-1][0], 'translation_stop_confirmed')
        self.assertGreaterEqual(events[-1][1]['stable_seconds'], .3)

    # 重复电调包、编码器持续爬动或较高轮速不能确认停稳。
    def test_duplicate_packets_or_creeping_encoders_cannot_confirm_stop(self):
        for repeated, rpm in ((True, 0), (False, 0), (False, 20)):
            cfg, clock = config(), [10.0]
            feedback = type('F', (), {
                'velocity_snapshot': lambda _, freshness: ((0, 0, 0), clock[0]),
                'esc_snapshot': lambda _, freshness: (((rpm,)*4,
                    (0 if repeated else int((clock[0]-10)*1000),)*4,
                    (10 if repeated else clock[0],)*4), clock[0])})()
            demo = M.AlignDemo(None, cfg, feedback, lambda *a, **k: None, None)
            with self.subTest(repeated=repeated, rpm=rpm), \
                    patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(M.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)), \
                    patch.object(demo, 'assert_distal'):
                with self.assertRaisesRegex(RuntimeError, 'encoders did not settle'):
                    demo.wait_stationary()

    # 电调缓存必须复制SDK数组，并拒绝超时反馈。
    def test_esc_feedback_is_copied_and_rejects_stale_data(self):
        feedback = M.Feedback()
        data = [[0]*4, [100]*4, [1]*4, [0]*4]
        with patch.object(M.time, 'monotonic', return_value=10):
            feedback.on_esc(data)
            data[1][0] = 200
            self.assertEqual(feedback.esc_snapshot(.25)[0][1][0], 100)
        with patch.object(M.time, 'monotonic', return_value=10.3):
            with self.assertRaisesRegex(RuntimeError, 'wheel feedback.*stale'):
                feedback.esc_snapshot(.25)

    # 内收后的位置扰动不能改变记录的前进轴，无可靠轴时限制回退。
    def test_return_uses_recorded_forward_axis_after_retract_drift(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        demo.start_position = (.00163, .00879)
        demo.approach_path = [dict(start=demo.start_position, heading=0, axis=(1, 0))]
        with patch.object(demo, 'position', return_value=(.00202, .01235)), \
                patch.object(demo, 'turn_to_offset'), patch.object(demo, 'translate') as move:
            demo.return_to_center()
            move.assert_not_called()  # Only .39mm along the real forward axis; do not reverse toward sideways drift.
        demo.approach_path[0]['axis'] = None
        with patch.object(demo, 'position', return_value=(.05, .00879)), patch.object(demo, 'turn_to_offset'):
            with self.assertRaisesRegex(RuntimeError, 'no reliable return direction'):
                demo.return_to_center()

    # 验证demo2的3cm回位边界，不能偷偷增加额外纠正动作。
    def test_return_accepts_three_cm_center_error_without_extra_correction(self):
        for residual, succeeds in ((.025, True), (.03, True), (.035, False)):
            demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
            demo.start_position = (0, 0)
            demo.approach_path = [dict(start=(residual, 0), heading=5, axis=(1, 0))]
            with self.subTest(residual=residual), patch.object(demo, 'position', return_value=(residual, 0)), \
                    patch.object(demo, 'turn_to_offset'), patch.object(demo, 'translate') as motion:
                if succeeds:
                    demo.return_to_center()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'return center error'):
                        demo.return_to_center()
                motion.assert_not_called()

    # 姿态设置可能扰动反馈，因此应在初态及停稳之后记录基准。
    def test_startup_baseline_is_recorded_after_initial_pose_and_stop(self):
        position, yaw = [(0, 0)], [25]
        feedback = type('F', (), {'yaw': lambda _: yaw[0]})()
        demo = M.AlignDemo(None, config(), feedback, lambda *a, **k: None, None)
        # 模拟初态设置带来的反馈位移，验证启动基准的记录时机。
        def initial(prefix):
            position[0], yaw[0] = (.002, .003), 30
        with patch.object(demo, 'lock_initial_distal'), patch.object(demo, 'initial_state', side_effect=initial), \
                patch.object(demo, 'wait_stationary'), patch.object(demo, 'position', side_effect=lambda: position[0]), \
                patch.object(demo, 'align_and_wait_for_grasp', side_effect=RuntimeError('test stops at baseline')):
            with self.assertRaisesRegex(RuntimeError, 'test stops at baseline'):
                demo.run()
        self.assertEqual(demo.start_yaw, 30)
        self.assertEqual(demo.start_position, (.002, .003))

    # 核对单轮只动正确舵机，且放置方向一直相对启动朝向。
    def test_cycle_preserves_joint_roles_and_anchors_drop_heading(self):
        cfg, events = config(), []
        # 把实机动作替换为事件记录，核对整轮舵机和朝向约束。
        class Demo(M.AlignDemo):
            # 记录锁定远端关节的模拟动作，不发送实机定位。
            def lock_initial_distal(self):
                events.append(('lock', self.arm['distal_servo_id'], self.arm['distal_hold_raw']))
            # 记录模拟初始姿态或按场景改变位置反馈，不控制实机。
            def initial_state(self, prefix):
                events.append((prefix, 'extended_open'))
            # 记录或模拟停稳步骤，不读取真实电调。
            def wait_stationary(self):
                events.append(('startup_stopped',))
            # 记录视觉对准和红外就绪步骤，不进行模型推理或抓取。
            def align_and_wait_for_grasp(self):
                events.append(('align_then_ir',))
            # 只记录抓夹开闭事件，不发送抓夹命令。
            def gripper(self, opened, stage):
                events.append((stage, opened))
            # 只记录底盘侧舵机的模拟目标，用于核对动作顺序。
            def move_base(self, target, stage):
                events.append((stage, self.arm['base_servo_id'], target))
            # 记录相对启动朝向的模拟转角。
            def turn_to_offset(self, target, stage):
                events.append((stage, target))
            # 返回预设模拟朝向，测试不订阅真实yaw。
            def heading_offset(self):
                return 0
            # 返回模拟位置，用于验证回位和启动中心判断。
            def position(self):
                return (0, 0)
            # 记录回位阶段，真实路径回退由其他测试单独覆盖。
            def return_to_center(self):
                events.append(('return_center',))
        feedback = type('F', (), {'yaw': lambda _: 25.0})()
        demo = Demo(None, cfg, feedback, lambda *args, **kwargs: None, None)
        demo.run()
        self.assertEqual(events, [('lock',2,1073),('start','extended_open'),('startup_stopped',),('align_then_ir',),
                         ('grasp_close',False),('carry_retract',1,1190),('return_center',),('turn_to_place',-90),
                         ('place_extend',1,600),('place_release',True),('return_retract',1,1190),
                         ('turn_to_start',0),('finish','extended_open')])


if __name__ == '__main__':
    unittest.main()
