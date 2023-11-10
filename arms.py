"""
Functions that deal with calibrating and moving the robot arm.
"""

from xarm.wrapper import XArmAPI
from utils import find_surface
import numpy as np

ip1 = '192.168.1.207'
ip2 = '192.168.1.213'

ips = {"photomaton": ip1, "big_drawing": ip2}

def get_ip(ip_index):
    return ips[ip_index]

def get_arm(ip):
    arm = XArmAPI(ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    return arm

def get_photomaton_arm():
    return get_arm(get_ip("photomaton"))

def get_big_drawing_arm():
    return get_arm(get_ip("big_drawing"))


def calibrate(arm, origin, p1, p2):
    Rx = 180
    Ry = 0
    Rz = 0

    calibrated_origin, _ = find_surface(arm, origin[0], origin[1], origin[2], Rx, Ry, Rz)[:3].reshape((1,-1))
    calibrated_p1, _ = find_surface(arm, p1[0], p1[1], p1[2], Rx, Ry, Rz)[:3].reshape((1,-1))
    calibrated_p2, _ = find_surface(arm, p2[0], p2[1], p2[2], Rx, Ry, Rz)[:3].reshape((1,-1))

    return calibrated_origin, calibrated_p1, calibrated_p2

def calibrate_from_dimensions(arm, origin, dx, dy):
    p1 = origin + np.array([dx, 0, 0])
    p2 = origin + np.array([0, dy, 0])
    return calibrate(arm, origin, p1, p2)

def draw_point_group(arm, point_group, dz=2, speed=100):
    """
    1. Go above the first point in the group.
    2. Lower the pen.
    3. Go through all of the points in the group.
    4. Raise the pen above the last point.

    Arguments:
    point_group -- an array of shape (3, number_of_points)
    dz -- the distance from the plane when the pen is lifted, in mm

    Returns the number of points drawn.
    """

    x = point_group[0, 0]
    y = point_group[1, 0]
    z = point_group[2, 0]
    
    arm.set_position(x=x, y=y, z=z+dz, roll=180, pitch=0, yaw=0, speed=speed, is_radian=0, wait=True, radius = None, relative = False)

    for i in range(0, point_group.shape[1]):
        
        x = point_group[0, i]
        y = point_group[1, i]
        z = point_group[2, i]
        arm.set_position_aa([x, y, z, 180, 0, 0], speed=speed, is_radian=0, wait=False, radius = None, relative = False, mvacc=2000)

    arm.set_position(x=0, y=0, z=2, roll=0, pitch=0, yaw=0, speed=speed, is_radian=0, wait=True, radius = None, relative = True)

    return len(point_group)

def draw_point_groups(arm, point_groups, dz=2, verbose=True, speed=100):
    """
    Draw the point groups in order.

    Arguments:
    point_groups -- a list of point groups, each point group being of shape (3, number_of_points)
    dz -- the distance from the plane when the pen is lifted, in mm (default 2.0)
    verbose -- if True, print the progress as a percentage
    """

    nb_points_drawn = 0
    nb_points = sum([len(group) for group in point_groups])
    for group in point_groups:
        nb_points_drawn += draw_point_group(arm, group, dz, speed)
        if verbose:
            print(f"{nb_points_drawn*100/nb_points}% complete...")