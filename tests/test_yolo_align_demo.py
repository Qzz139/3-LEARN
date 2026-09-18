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
    def test_image_left_turns_left_and_right_turns_right_with_limited_steps(self):
        a = config()['alignment']
        self.assertEqual(M.correction_degrees(M.pixel_error(box(320), .5), a), 5)
        self.assertEqual(M.correction_degrees(M.pixel_error(box(960), .5), a), -5)
        self.assertEqual(M.correction_degrees(M.pixel_error(box(650), .5), a), 0)
        self.assertGreater(M.correction_degrees(-.03, a), 0)

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
                patch.object(demo, 'infrared_confirmed', side_effect=[False, True]) as infrared, \
                patch.object(M.time, 'sleep'):
            demo.align_and_wait_for_grasp()
        self.assertEqual(infrared.call_count, 2)
        self.assertEqual(events.count('alignment_stable'), 4)
        self.assertEqual(events[-1], 'alignment_and_distance_ready')

    def test_cycle_preserves_joint_roles_and_anchors_drop_heading(self):
        cfg, events = config(), []
        class Demo(M.AlignDemo):
            def lock_initial_distal(self):
                events.append(('lock', self.arm['distal_servo_id'], self.arm['distal_hold_raw']))
            def initial_state(self, prefix):
                events.append((prefix, 'extended_open'))
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
        feedback = type('F', (), {'yaw': lambda _: 25.0})()
        demo = Demo(None, cfg, feedback, lambda *args, **kwargs: None, None)
        demo.run()
        self.assertEqual(events, [('lock',2,1073),('start','extended_open'),('align_then_ir',),
                         ('grasp_close',False),('carry_retract',1,1190),('turn_to_place',-90),
                         ('place_extend',1,600),('place_release',True),('return_retract',1,1190),
                         ('turn_to_start',0),('finish','extended_open')])


if __name__ == '__main__':
    unittest.main()
