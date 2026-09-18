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


def config():
    return M.validate_alignment(json.loads((ROOT / 'config/yolo_align_pick_place.json').read_text()))


def box(cx, cls=32, width=1280, height=720, scale=1):
    return dict(class_id=cls, confidence=.9, xyxy=[cx-60*scale, 300, cx+60*scale, 420],
                image_width=width, image_height=height)


class TestAlignment(unittest.TestCase):
    def test_camera_feed_drains_while_control_is_idle_and_returns_a_new_frame(self):
        class Camera:
            count = 0
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

    def test_target_selection_excludes_bottles_and_tracks_one_ball(self):
        a = config()['alignment']
        previous = box(320)
        bottle, other_ball, same_ball = box(640, 39), box(1000), box(380)
        self.assertEqual(M.choose_target([bottle, same_ball, other_ball], a), same_ball)
        self.assertEqual(M.choose_target([other_ball, same_ball], a, previous), same_ball)
        self.assertIsNone(M.choose_target([other_ball], a, previous))
        self.assertIsNone(M.choose_target([box(340, scale=10)], a, previous))

    def test_heading_wrap_and_place_compensation_use_startup_reference(self):
        cfg = config()
        feedback = type('F', (), {'yaw': lambda self: 160.0})()
        demo = M.AlignDemo(None, cfg, feedback, lambda *args, **kwargs: None, None)
        demo.start_yaw = -170
        self.assertEqual(demo.heading_offset(), 30)
        # Aligned 30deg left -> startup-right placement needs -120deg total, not -90.
        self.assertEqual(M.wrap_degrees(-90 - demo.heading_offset()), -120)

    def test_measured_yaw_decreases_for_positive_sdk_turn_on_this_ep(self):
        cfg = config()
        feedback = type('F', (), {'yaw': lambda _: .87})()
        demo = M.AlignDemo(None, cfg, feedback, lambda *args, **kwargs: None, None)
        demo.start_yaw = 6.04
        self.assertAlmostEqual(demo.heading_offset(), 5.17)

    def test_search_offsets_stay_within_configured_range(self):
        self.assertEqual(M.search_offsets(config()['alignment']), [15, -15, 30, -30, 45, -45, 60, -60])

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

    def test_target_loss_stops_instead_of_switching_to_another_ball(self):
        cfg = config()
        class Detector:
            frames = iter([[box(640)], [box(1100)], [box(1100)], [box(1100)]])
            def detect(self, _):
                return next(self.frames), .01
        demo = M.AlignDemo(type('Bot', (), {'camera': None})(), cfg, None, lambda *args, **kwargs: None, Detector())
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'heading_offset', return_value=0), patch.object(M.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'locked target lost'):
                demo.align_and_wait_for_grasp()

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

    def test_successful_action_without_yaw_progress_is_rejected(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        with patch.object(demo, 'heading_offset', side_effect=[0, .02]), \
                patch.object(demo, 'turn'), patch.object(M.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'without actual yaw progress'):
                demo.align_turn(5)

    def test_unstable_near_infrared_never_advances(self):
        cfg = config()
        detector = type('Detector', (), {'detect': lambda *_: ([box(640)], .01)})()
        demo = M.AlignDemo(type('Bot', (), {'camera': None})(), cfg, None, lambda *a, **k: None, detector)
        with patch.object(demo, 'assert_distal'), patch.object(demo, 'heading_offset', return_value=0), \
                patch.object(demo, 'infrared_state', side_effect=['wait', 'ready']), \
                patch.object(demo, 'approach_step') as advance, patch.object(M.time, 'sleep'):
            demo.align_and_wait_for_grasp()
        advance.assert_not_called()

    def test_infrared_uses_distinct_new_samples_and_rejects_invalid_data(self):
        cfg = config()
        for values, expected in (([15, 15, 15], 'ready'), ([90, 90, 90], 'far'),
                                 ([15, 90, 15, 90], 'wait'), ([0], 'invalid')):
            clock = [10.0]
            feedback = M.Feedback()
            feedback.distance, feedback.distance_time = (90,), 9.99
            sequence = iter(values)
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

    def test_maximum_travel_prevents_another_forward_command(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        demo.travel_m = .2
        with patch.object(demo, 'translate') as move:
            with self.assertRaisesRegex(RuntimeError, 'maximum approach'):
                demo.approach_step()
            move.assert_not_called()

    def test_odometry_controls_slow_motion_and_retreats_to_waypoint(self):
        cfg, clock, position, velocity, commands = config(), [10.0], [0.0, 0.0], [0.0], []
        def drive_speed(x, y, z, timeout):
            velocity[0] = x
            commands.append(x)
            return False  # Actual official no-ACK SDK behavior, not a rejection.
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

    def test_retreat_can_pass_waypoint_with_small_lateral_error(self):
        cfg, clock, position, velocity = config(), [10.0], [.02, 0.0], [0.0]
        def drive_speed(x, y, z, timeout):
            velocity[0] = x
            return False
        def drive_wheels(**kw):
            velocity[0] = 0
            return True
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

    def test_threshold_or_sensor_failure_stops_velocity_segment(self):
        for readings, raises in (([10], False), ([90, RuntimeError('sensor failed')], True)):
            cfg, clock, position, velocity, commands = config(), [10.0], [0.0, 0.0], [0.0], []
            def drive_speed(x, y, z, timeout):
                velocity[0] = x
                commands.append(x)
                return True
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

    def test_return_replays_headings_in_reverse_before_checking_center(self):
        demo = M.AlignDemo(None, config(), None, lambda *a, **k: None, None)
        demo.start_position = (0, 0)
        demo.approach_path = [dict(start=(0, 0), heading=20), dict(start=(.005, 0), heading=25)]
        position = [( .01, 0)]
        def retreat(distance, stage, return_target):
            position[0] = return_target
        with patch.object(demo, 'position', side_effect=lambda: position[0]), \
                patch.object(demo, 'translate', side_effect=retreat) as move, \
                patch.object(demo, 'turn_to_offset') as turn:
            demo.return_to_center()
        self.assertEqual([c[0][0] for c in turn.call_args_list], [25, 20])
        self.assertEqual([c[1]['return_target'] for c in move.call_args_list], [(.005, 0), (0, 0)])

    def test_stalled_odometry_stops_chassis(self):
        cfg, clock, commands = config(), [10.0], []
        def drive_speed(x, y, z, timeout):
            commands.append(x)
            return True
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

    def test_acknowledged_wheel_stop_brakes_when_velocity_stop_is_ignored(self):
        velocity, requests = [.04], []
        def drive_speed(**kw):
            return False
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

    def test_startup_baseline_is_recorded_after_initial_pose_and_stop(self):
        position, yaw = [(0, 0)], [25]
        feedback = type('F', (), {'yaw': lambda _: yaw[0]})()
        demo = M.AlignDemo(None, config(), feedback, lambda *a, **k: None, None)
        def initial(prefix):
            position[0], yaw[0] = (.002, .003), 30
        with patch.object(demo, 'lock_initial_distal'), patch.object(demo, 'initial_state', side_effect=initial), \
                patch.object(demo, 'wait_stationary'), patch.object(demo, 'position', side_effect=lambda: position[0]), \
                patch.object(demo, 'align_and_wait_for_grasp', side_effect=RuntimeError('test stops at baseline')):
            with self.assertRaisesRegex(RuntimeError, 'test stops at baseline'):
                demo.run()
        self.assertEqual(demo.start_yaw, 30)
        self.assertEqual(demo.start_position, (.002, .003))

    def test_cycle_preserves_joint_roles_and_anchors_drop_heading(self):
        cfg, events = config(), []
        class Demo(M.AlignDemo):
            def lock_initial_distal(self):
                events.append(('lock', self.arm['distal_servo_id'], self.arm['distal_hold_raw']))
            def initial_state(self, prefix):
                events.append((prefix, 'extended_open'))
            def wait_stationary(self):
                events.append(('startup_stopped',))
            def align_and_wait_for_grasp(self):
                events.append(('align_then_ir',))
            def gripper(self, opened, stage):
                events.append((stage, opened))
            def move_base(self, target, stage):
                events.append((stage, self.arm['base_servo_id'], target))
            def turn_to_offset(self, target, stage):
                events.append((stage, target))
            def heading_offset(self):
                return 0
            def position(self):
                return (0, 0)
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
