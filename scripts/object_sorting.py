#!/usr/bin/env python3
# 六件分拣：按当前朝向选择最近的待取方位，球右放、瓶左放，各使用三个位置。
"""Sort balls and bottles from configured directions into three slots per class."""
import math
import time

from sorting_state_machine import SortingStateMachine
import yolo_align_pick_place_demo as align


CLASS_NAMES = {32: 'ball', 39: 'bottle'}


def nearest_pending_direction(directions, current_heading):
    return min((d for d in directions if d['status'] == 'pending'),
               key=lambda d: abs(align.wrap_degrees(d['heading'] - current_heading)),
               default=None)


# 复用demo2参数校验，再核对六个抓取方位及每类三个放置角度。
def validate_sorting(cfg):
    align.validate_alignment(cfg)
    if set(cfg['alignment']['target_class_ids']) != set(CLASS_NAMES):
        raise ValueError('sorting requires both ball (32) and bottle (39); omit --target')
    s = cfg['sorting']
    for key in ('pick_headings_degrees', 'sector_scan_offsets_degrees'):
        values = s[key]
        if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                             or not math.isfinite(v) or not -180 <= v <= 180 for v in values):
            raise ValueError(key + ' must contain finite headings within [-180, 180]')
        if len({align.wrap_degrees(v) for v in values}) != len(values):
            raise ValueError(key + ' contains duplicate directions')
    limit = cfg['alignment']['search_limit_degrees']
    if any(abs(v) > limit for v in s['sector_scan_offsets_degrees']):
        raise ValueError('sector scans must stay within alignment.search_limit_degrees')
    for name in CLASS_NAMES.values():
        slots = s['placement_headings_degrees'][name]
        if len(slots) != 3:
            raise ValueError(name + ' must have exactly three placement headings')
        defined = [v for v in slots if v is not None]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not -180 < v < 180 for v in defined):
            raise ValueError('placement headings must be finite and within (-180, 180)')
        wrong_side = any(v >= 0 for v in defined) if name == 'ball' else any(v <= 0 for v in defined)
        if wrong_side:
            raise ValueError('ball slots must be right (negative); bottle slots left (positive)')
        if len(set(defined)) != len(defined):
            raise ValueError(name + ' placement headings must be distinct')
    return cfg


# 实机执行前要求六个放置角度都已填写；预览允许保留空位置。
def require_placement_slots(cfg):
    if any(v is None for slots in cfg['sorting']['placement_headings_degrees'].values() for v in slots):
        raise ValueError('fill the six placement headings in config/object_sorting.json before executing')


# 稳定版按配置顺序处理方位，复用视觉对准、红外接近和路径回退。
class SortingDemo(align.AlignDemo):
    """Keep the starting center and heading as the reference for every object."""

    def __init__(self, bot, cfg, feedback, record, detector, state_machine=None):
        super().__init__(bot, cfg, feedback, record, detector)
        self.state_machine = state_machine

    # 内收转到预定角度后外伸识别；无目标时左右扫描，不直接换锁定目标。
    def find_target(self, heading, available_classes):
        selection = dict(self.cfg['alignment'], target_class_ids=available_classes)
        for offset in self.cfg['sorting']['sector_scan_offsets_degrees']:
            target_heading = align.wrap_degrees(heading + offset)
            # 已在目标朝向时不转底盘，也无需为这次零角度转向先内收。
            if abs(align.wrap_degrees(target_heading - self.heading_offset())) > selection['heading_tolerance_degrees']:
                self.move_base(self.arm['base_retracted_raw'], 'search_retract')
                self.turn_to_offset(target_heading, 'sector_scan_turn')
            self.move_base(self.arm['base_extended_raw'], 'search_extend')
            # 两帧新图像均无目标才扫描下一方向，过期图像不计数。
            deadline, empty = time.monotonic() + 5.0, 0
            while time.monotonic() < deadline and empty < 2:
                self.assert_distal()
                detections, age = self.detector.detect(self.bot.camera)
                if age > selection['max_result_age_s']:
                    continue
                target = align.choose_target(detections, selection)
                # 优先取离抓取轴最近的目标；中心±15%外的目标留给相邻取物方向。
                off_axis = (target is not None and
                            abs(align.pixel_error(target, selection['axis_x_ratio'])) > .15)
                self.record('sector_observed', heading=heading, scan_offset=offset,
                            target=target, age_s=age, ignored_off_axis=off_axis)
                if target and not off_axis:
                    return target
                empty += 1
            if empty < 2:
                raise RuntimeError('no fresh camera results during sector scan')
        return None

    # 整轮共享启动中心与朝向，两类分别计数并按顺序占用各自三个放置位。
    def run(self):
        require_placement_slots(self.cfg)
        machine = self.state_machine or SortingStateMachine.from_file(
            align.ROOT / 'config/sorting_state_machine.yaml')
        counts = {32: 0, 39: 0}

        def advance(event):
            source, _, target_state = machine.advance(event)
            self.record('sorting_state_transition', source=source, event=event, target=target_state)

        try:
            while machine.current not in machine.terminal:
                state = machine.current
                if state == 'initialize':
                    self.lock_initial_distal()
                    self.initial_state('start')
                    self.wait_stationary()
                    self.start_yaw = self.feedback.yaw()
                    self.start_position = self.position()
                    self.pick_directions = [dict(heading=heading, status='pending')
                                            for heading in self.cfg['sorting']['pick_headings_degrees']]
                    self.record('sorting_started', start_yaw=self.start_yaw,
                                start_position=self.start_position,
                                directions=[dict(d) for d in self.pick_directions])
                    advance('ready')
                elif state == 'select_direction':
                    available = [class_id for class_id in CLASS_NAMES if counts[class_id] < 3]
                    current_heading = self.heading_offset()
                    direction = nearest_pending_direction(self.pick_directions, current_heading) if available else None
                    if direction is None:
                        advance('complete')
                        continue
                    heading = direction['heading']
                    self.record('next_pick_direction', heading=heading, current_heading=current_heading,
                                turn_degrees=align.wrap_degrees(heading - current_heading),
                                directions=[dict(d) for d in self.pick_directions])
                    advance('target')
                elif state == 'search':
                    target = self.find_target(heading, available)
                    if target is None:
                        direction['status'] = 'not_found'
                        self.record('sector_empty', heading=heading,
                                    directions=[dict(d) for d in self.pick_directions])
                        advance('empty')
                    else:
                        advance('found')
                elif state == 'align':
                    self.approach_path, self.travel_m = [], 0.0
                    target = self.align_and_wait_for_grasp(initial_target=target, search_center=heading)
                    class_id = target['class_id']
                    slot = counts[class_id]
                    place_heading = self.cfg['sorting']['placement_headings_degrees'][CLASS_NAMES[class_id]][slot]
                    self.record('sorting_grasp', class_id=class_id, slot=slot + 1,
                                place_heading=place_heading)
                    advance('ready')
                elif state == 'grasp':
                    self.gripper(False, 'grasp_close')
                    self.move_base(self.arm['base_retracted_raw'], 'carry_retract')
                    advance('held')
                elif state == 'return_to_center':
                    self.return_to_center()
                    advance('centered')
                elif state == 'place':
                    self.turn_to_offset(place_heading, 'turn_to_place')
                    self.move_base(self.arm['base_extended_raw'], 'place_extend')
                    self.gripper(True, 'place_release')
                    self.move_base(self.arm['base_retracted_raw'], 'after_place_retract')
                    self.wait_stationary()
                    error = align.position_distance(self.position(), self.start_position)
                    if error > self.cfg['approach']['center_tolerance_m']:
                        raise RuntimeError('placement shifted chassis outside starting center tolerance')
                    counts[class_id] += 1
                    direction['status'] = 'taken'
                    self.record('sorting_item_placed', class_id=class_id, slot=slot + 1,
                                counts=counts.copy(), center_error_m=error, pick_heading=heading,
                                directions=[dict(d) for d in self.pick_directions])
                    advance('released')
                elif state == 'finish':
                    self.move_base(self.arm['base_retracted_raw'], 'finish_retract')
                    self.turn_to_offset(0, 'finish_turn_to_start')
                    self.initial_state('finish')
                    self.wait_stationary()
                    advance('done')
                    self.record('sorting_complete', counts=counts, all_six_placed=sum(counts.values()) == 6,
                                directions=[dict(d) for d in self.pick_directions],
                                arm='extended', gripper='open')
                else:
                    raise RuntimeError('state has no implementation: ' + state)
        except BaseException:
            if machine.current not in machine.terminal:
                advance('error')
            raise


if __name__ == '__main__':
    raise SystemExit(align.main(
        demo_class=SortingDemo,
        default_config=align.ROOT / 'config/object_sorting.json',
        validator=validate_sorting,
        execution_check=require_placement_slots,
        preview='预览：选择转角最小的待取方位 → 内收转向再外伸 → YOLO锁定球或瓶 → 对准接近抓取 → 内收退回中心 → 分类放入下一空位 → 记录方位已取 → 松爪内收转到最近待取方位再外伸；最终内收回正并恢复外伸松爪。',
    ))
