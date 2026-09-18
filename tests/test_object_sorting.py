import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import object_sorting as sorting


def config():
    return sorting.validate_sorting(json.loads((ROOT / 'config/object_sorting.json').read_text()))


class SortingTests(unittest.TestCase):
    def test_full_cycle_keeps_reference_and_uses_each_slot_once(self):
        cfg = config()
        events = []
        demo = sorting.SortingDemo(None, cfg, SimpleNamespace(yaw=lambda: 12.0),
                                  lambda stage, **data: events.append((stage, data)), None)
        demo.lock_initial_distal = Mock()
        demo.initial_state = Mock()
        demo.wait_stationary = Mock()
        demo.position = Mock(return_value=(0.0, 0.0))
        demo.turn_to_offset = Mock()
        demo.gripper = Mock()
        demo.move_base = Mock()
        demo.return_to_center = Mock()
        classes = iter([32, 32, 32, 39, 39, 39])
        demo.find_target = Mock(side_effect=lambda heading, available: {'class_id': next(classes)})

        def align_target(initial_target, search_center):
            self.assertEqual(demo.approach_path, [])
            self.assertEqual(demo.travel_m, 0)
            self.assertEqual(demo.start_yaw, 12.0)
            demo.approach_path.append('old segment')
            demo.travel_m = .05
            return initial_target

        demo.align_and_wait_for_grasp = Mock(side_effect=align_target)
        demo.run()
        demo.lock_initial_distal.assert_called_once()
        self.assertEqual(demo.return_to_center.call_count, 6)
        self.assertEqual([call[0][0] for call in demo.turn_to_offset.call_args_list],
                         [-90, -72, -108, 90, 72, 108, 0])
        self.assertEqual([call[0][0] for call in demo.find_target.call_args_list],
                         cfg['sorting']['pick_headings_degrees'])
        self.assertEqual(demo.find_target.call_args_list[3][0][1], [39])
        self.assertEqual(demo.initial_state.call_args_list[0][0], ('start',))
        self.assertEqual(demo.initial_state.call_args_list[1][0], ('finish',))
        self.assertEqual(events[-1][1]['counts'], {32: 3, 39: 3})
        self.assertTrue(events[-1][1]['all_six_placed'])
        self.assertEqual(sum(call[0][1] == 'after_place_retract'
                             for call in demo.move_base.call_args_list), 6)
        self.assertEqual(demo.move_base.call_args_list[-1][0], (1190, 'finish_retract'))

    def test_empty_sector_does_not_consume_slot_or_claim_six_done(self):
        cfg = config()
        events = []
        demo = sorting.SortingDemo(None, cfg, SimpleNamespace(yaw=lambda: 0),
                                  lambda stage, **data: events.append((stage, data)), None)
        for name in ('lock_initial_distal', 'initial_state', 'wait_stationary', 'turn_to_offset', 'move_base'):
            setattr(demo, name, Mock())
        demo.position = Mock(return_value=(0, 0))
        demo.find_target = Mock(return_value=None)
        demo.gripper = Mock()
        demo.run()
        demo.gripper.assert_not_called()
        self.assertFalse(events[-1][1]['all_six_placed'])
        self.assertEqual(events[-1][1]['counts'], {32: 0, 39: 0})

    def test_each_search_turn_retracts_before_turn_and_extends_before_detection(self):
        cfg = config()
        target = dict(class_id=39, confidence=.9, xyxy=[420, 50, 540, 450],
                      image_width=960, image_height=540)
        actions = []
        frames = iter([[], [], [target]])

        def detect(_camera):
            actions.append(('detect',))
            return next(frames), .1

        demo = sorting.SortingDemo(SimpleNamespace(camera=None), cfg, None, Mock(),
                                  SimpleNamespace(detect=detect))
        demo.assert_distal = Mock()
        demo.move_base = lambda raw, stage: actions.append(('arm', raw))
        demo.turn_to_offset = lambda heading, stage: actions.append(('turn', heading))
        self.assertEqual(demo.find_target(36, [32, 39]), target)
        self.assertEqual(actions, [('arm', 1190), ('turn', 36), ('arm', 600),
                                   ('detect',), ('detect',),
                                   ('arm', 1190), ('turn', 51), ('arm', 600), ('detect',)])

    def test_wrong_side_duplicate_and_missing_slots_are_rejected(self):
        for slots in ([90, -72, -108], [-90, -90, -108]):
            cfg = config()
            cfg['sorting']['placement_headings_degrees']['ball'] = slots
            with self.assertRaises(ValueError):
                sorting.validate_sorting(cfg)
        cfg = config()
        cfg['sorting']['placement_headings_degrees']['ball'][0] = None
        sorting.validate_sorting(cfg)
        with self.assertRaises(ValueError):
            sorting.require_placement_slots(cfg)

    def test_rear_alignment_does_not_use_front_sector_bounds(self):
        cfg = config()
        target = dict(class_id=32, confidence=.9, xyxy=[480, 200, 580, 300],
                      image_width=960, image_height=540)
        detector = SimpleNamespace(detect=Mock(return_value=([target], .1)))
        demo = sorting.SortingDemo(SimpleNamespace(camera=None), cfg, None, Mock(), detector)
        demo.assert_distal = Mock()
        demo.heading_offset = Mock(return_value=179)
        demo.align_turn = Mock(side_effect=RuntimeError('correction attempted'))
        with self.assertRaisesRegex(RuntimeError, 'correction attempted'):
            demo.align_and_wait_for_grasp(initial_target=target, search_center=180)
        demo.align_turn.assert_called_once_with(-5.0)


if __name__ == '__main__':
    unittest.main()
