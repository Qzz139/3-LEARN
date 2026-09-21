"""Small, ROS-independent helpers for the simulated six-direction task."""
import math


PICK_HEADINGS = (0, 36, -36, 144, -144, 180)
PICK_OBJECTS = {0: ('ball_1', 32), 36: ('bottle_2', 39),
                -36: ('ball_3', 32), 144: ('bottle_4', 39),
                -144: ('ball_5', 32), 180: ('bottle_6', 39)}
PLACE_HEADINGS = {'ball': (-90, -72, -108), 'bottle': (90, 72, 108)}


def wrap_radians(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def nearest_heading(pending, current_degrees):
    if not pending:
        return None
    return min(pending, key=lambda heading: abs(wrap_radians(
        math.radians(heading - current_degrees))))


def yaw_from_quaternion(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def limited_turn(error):
    if abs(error) < math.radians(2.5):
        return 0.0
    return math.copysign(min(0.38, max(0.12, 0.9 * abs(error))), error)
