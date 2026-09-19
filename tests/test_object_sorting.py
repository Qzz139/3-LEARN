# 纯软件回归测试：使用模拟模型、时钟及SDK接口，不发送实机动作。
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import object_sorting as sorting


# 读取仓库配置，确保测试使用当前稳定版参数。
def config():
    return sorting.validate_sorting(json.loads((ROOT / 'config/object_sorting.json').read_text()))


# 验证稳定版多物品调度、放置位分配和搜索姿态顺序。
class SortingTests(unittest.TestCase):
    # 核对六件共享启动基准、两类各三个位及逐件路径状态重置。
    def test_full_cycle_keeps_reference_and_uses_each_slot_once(self):
        cfg = config()
        events = []
        demo = sorting.SortingDemo(None, cfg, SimpleNamespace(yaw=lambda: 12.0),
                                  lambda stage, **data: events.append((stage, data)), None)
        demo.lock_initial_distal = Mock()
        demo.initial_state = Mock()
        demo.wait_stationary = Mock()
        demo.position = Mock(return_value=(0.0, 0.0))
        heading = [0.0]
        demo.heading_offset = Mock(side_effect=lambda: heading[0])
        demo.turn_to_offset = Mock(side_effect=lambda value, stage: heading.__setitem__(0, value))
        demo.gripper = Mock()
        demo.move_base = Mock()
        demo.return_to_center = Mock()
        classes = {0: 32, -36: 39, 36: 39, 144: 32, -144: 32, 180: 39}

        def find_target(value, available):
            heading[0] = value
            self.assertIn(classes[value], available)
            return {'class_id': classes[value]}

        demo.find_target = Mock(side_effect=find_target)

        # 校验逐件路径状态已重置和启动基准不变，再生成模拟接近记录。
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
                         [-90, 90, 72, -72, -108, 108, 0])
        self.assertEqual([call[0][0] for call in demo.find_target.call_args_list],
                         [0, -36, 36, 144, -144, 180])
        self.assertEqual(demo.find_target.call_args_list[5][0][1], [39])
        self.assertEqual(demo.initial_state.call_args_list[0][0], ('start',))
        self.assertEqual(demo.initial_state.call_args_list[1][0], ('finish',))
        self.assertEqual(events[-1][1]['counts'], {32: 3, 39: 3})
        self.assertTrue(events[-1][1]['all_six_placed'])
        self.assertTrue(all(d['status'] == 'taken' for d in events[-1][1]['directions']))
        self.assertTrue(all(d['status'] == 'pending' for d in events[0][1]['directions']))
        self.assertEqual(sum(call[0][1] == 'after_place_retract'
                             for call in demo.move_base.call_args_list), 6)
        self.assertEqual(demo.move_base.call_args_list[-1][0], (1190, 'finish_retract'))

    # 空方向不夹爪、不占用放置位，也不能宣称已处理六件。
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
        self.assertTrue(all(d['status'] == 'not_found' for d in events[-1][1]['directions']))
        self.assertEqual(demo.find_target.call_count, 6)

    def test_nearest_direction_wraps_at_180_and_excludes_taken_and_missing(self):
        directions = [dict(heading=178, status='taken'),
                      dict(heading=176, status='not_found'),
                      dict(heading=150, status='pending'),
                      dict(heading=-170, status='pending')]
        self.assertIs(sorting.nearest_pending_direction(directions, 179), directions[3])
        for d in directions:
            d['status'] = 'taken'
        self.assertIsNone(sorting.nearest_pending_direction(directions, 179))

    # 记录模拟动作顺序，核对每次搜索转向前内收、取图前外伸。
    def test_each_search_turn_retracts_before_turn_and_extends_before_detection(self):
        cfg = config()
        target = dict(class_id=39, confidence=.9, xyxy=[420, 50, 540, 450],
                      image_width=960, image_height=540)
        actions = []
        frames = iter([[], [], [target]])

        # 按预设次序返回图像识别结果，用于核对搜索和动作先后。
        def detect(_camera):
            actions.append(('detect',))
            return next(frames), .1

        demo = sorting.SortingDemo(SimpleNamespace(camera=None), cfg, None, Mock(),
                                  SimpleNamespace(detect=detect))
        demo.assert_distal = Mock()
        demo.move_base = lambda raw, stage: actions.append(('arm', raw))
        heading = [0]
        demo.heading_offset = lambda: heading[0]
        def turn_to_offset(value, _stage):
            actions.append(('turn', value))
            heading[0] = value
        demo.turn_to_offset = turn_to_offset
        self.assertEqual(demo.find_target(36, [32, 39]), target)
        self.assertEqual(actions, [('arm', 1190), ('turn', 36), ('arm', 600),
                                   ('detect',), ('detect',),
                                   ('arm', 1190), ('turn', 51), ('arm', 600), ('detect',)])

    def test_sector_skips_edge_target_until_scan_brings_it_nearer_axis(self):
        cfg = config()
        edge = dict(class_id=32, confidence=.9, xyxy=[625, 343, 769, 485],
                    image_width=960, image_height=540)
        centered = dict(edge, xyxy=[340, 96, 460, 451])
        frames = iter([[edge], [edge], [centered]])
        events, turns = [], []
        demo = sorting.SortingDemo(SimpleNamespace(camera=None), cfg, None,
                                  lambda stage, **data: events.append((stage, data)),
                                  SimpleNamespace(detect=lambda _camera: (next(frames), .1)))
        demo.assert_distal = Mock()
        demo.move_base = Mock()
        heading = [0]
        demo.heading_offset = lambda: heading[0]
        def turn_to_offset(value, _stage):
            turns.append(value)
            heading[0] = value
        demo.turn_to_offset = turn_to_offset
        self.assertEqual(demo.find_target(0, [32, 39]), centered)
        self.assertEqual(turns, [15])
        self.assertTrue(events[0][1]['ignored_off_axis'])
        self.assertFalse(events[-1][1]['ignored_off_axis'])

    def test_starting_direction_zero_does_not_retract_before_looking(self):
        cfg = config()
        target = dict(class_id=32, confidence=.9, xyxy=[430, 320, 530, 420],
                      image_width=960, image_height=540)
        demo = sorting.SortingDemo(SimpleNamespace(camera=None), cfg, None, Mock(),
                                  SimpleNamespace(detect=lambda _camera: ([target], .1)))
        demo.assert_distal = Mock()
        demo.heading_offset = Mock(return_value=0)
        demo.move_base = Mock()
        demo.turn_to_offset = Mock()
        self.assertEqual(demo.find_target(0, [32, 39]), target)
        demo.turn_to_offset.assert_not_called()
        demo.move_base.assert_called_once_with(600, 'search_extend')

    def test_place_heading_accepts_small_unactionable_residual(self):
        cfg = config()
        self.assertEqual(cfg['alignment']['heading_tolerance_degrees'], 3.0)
        demo = sorting.SortingDemo(None, cfg, None, Mock(), None)
        demo.heading_offset = Mock(return_value=87.5)
        demo.turn = Mock()
        demo.turn_to_offset(90, 'turn_to_place')
        demo.turn.assert_not_called()

    # 放置位不能落到错误侧或重复，缺少位置时不能进入实机流程。
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

    # 后方对准以当前扇区为基准，不能误用正前方的角度限制。
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
