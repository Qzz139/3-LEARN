"""Check the simulation layout and shortest-direction behavior without ROS/Gazebo."""
import math
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / 'ros2_ws/src/ep_sorting_sim'
sys.path.insert(0, str(SIM))
from ep_sorting_sim.geometry import PICK_OBJECTS, nearest_heading, wrap_radians  # noqa: E402


class GazeboSimulationTests(unittest.TestCase):
    def test_world_contains_six_pickups_and_six_places(self):
        world = ET.parse(SIM / 'worlds/sorting.world').getroot()
        names = [node.attrib['name'] for node in world.findall('./world/model')]
        self.assertEqual(sum(name.startswith('ball_') and 'zone' not in name for name in names), 3)
        self.assertEqual(sum(name.startswith('bottle_') and 'zone' not in name for name in names), 3)
        self.assertEqual(sum('zone' in name for name in names), 6)
        robot = world.find('./world/model[@name="ep_sim"]')
        self.assertIsNotNone(robot.find('./plugin[@name="drive"]'))
        grippers = robot.findall('./plugin[@filename="libgazebo_ros_vacuum_gripper.so"]')
        self.assertEqual(len(grippers), 6)
        for gripper in grippers:
            target = gripper.attrib['name'][len('grip_'):]
            excluded = {node.text for node in gripper.findall('fixed')}
            self.assertNotIn(target, excluded)
            self.assertTrue({name for name, _class_id in PICK_OBJECTS.values() if name != target}
                            <= excluded)
        arm = robot.find('./link[@name="arm_1_link"]')
        self.assertIsNotNone(arm.find('./sensor[@name="camera"]'))
        self.assertIsNotNone(arm.find('./sensor[@name="ir"]'))
        self.assertIsNotNone(robot.find('./joint[@name="arm_1_joint"]'))
        self.assertIsNotNone(robot.find('./link[@name="gripper_tip_link"]'))
        self.assertIsNotNone(robot.find('./link[@name="base_link"]//uri'))

    def test_shortest_pending_heading_wraps_across_back(self):
        self.assertEqual(nearest_heading([0, -144, 180], 175), 180)
        self.assertEqual(nearest_heading([-144, 36], 170), -144)
        self.assertAlmostEqual(wrap_radians(math.radians(181)), math.radians(-179))

    def test_retracting_base_joint_lifts_and_moves_gripper_inward(self):
        robot = ET.parse(SIM / 'worlds/sorting.world').getroot().find(
            './world/model[@name="ep_sim"]')
        pivot = [float(v) for v in robot.find('./joint[@name="arm_1_joint"]/pose').text.split()]
        tip = [float(v) for v in robot.find('./link[@name="arm_1_link"]/sensor[@name="ir"]/pose').text.split()]

        def position(angle):
            dx, dz = tip[0], tip[2]  # Sensor pose is already relative to the arm pivot.
            x = pivot[0] + dx * math.cos(angle) + dz * math.sin(angle)
            z = pivot[2] - dx * math.sin(angle) + dz * math.cos(angle)
            return x, z

        extended, retracted = position(0), position(-1.3)
        self.assertLess(retracted[0], extended[0])
        self.assertGreater(retracted[1], extended[1])


if __name__ == '__main__':
    unittest.main()
