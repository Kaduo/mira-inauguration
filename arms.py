"""
Functions that deal with calibrating and moving the robot arm.
"""

from xarm.wrapper import XArmAPI
from utils import find_surface
import numpy as np
from time import time

ip1 = "192.168.1.207"
ip2 = "192.168.1.213"

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
    arm = get_arm(get_ip("photomaton"))
    enable_arm(arm)
    return arm


def get_big_drawing_arm():
    arm = get_arm(get_ip("big_drawing"))
    enable_arm(arm)
    return arm


def enable_arm(arm):
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)


def calibrate(arm, points, Rx=180, Ry=0, Rz=0, relative_epsilon=None, absolute_epsilon=1):
    """
    Return a list of the calibrated coordinates, with the origin in first.
    """
    res = []

    try:
        if relative_epsilon is None:
            new_relative_epsilon = [0] * len(points)
        else:
            new_relative_epsilon = relative_epsilon
        for re, ae, p in zip(new_relative_epsilon, absolute_epsilon, points):
            x, y, z = p
            res.append(
                np.array(
                    find_surface(
                        arm, x, y, z, Rx, Ry, Rz, relative_epsilon=re, absolute_epsilon=ae
                    )[0][:3]
                ).reshape((1, -1))
            )

    except TypeError:
        print("what ??")
        if relative_epsilon is None:
            new_relative_epsilon = 0
        else:
            new_relative_epsilon = relative_epsilon
        for p in points:
            x, y, z = p
            res.append(
                np.array(
                    find_surface(
                        arm,
                        x,
                        y,
                        z,
                        Rx,
                        Ry,
                        Rz,
                        relative_epsilon=new_relative_epsilon,
                        absolute_epsilon=absolute_epsilon,
                    )[0][:3]
                ).reshape((1, -1))
            )

    return res


def calibrate_from_dimensions(arm, origin, dx, dy):
    p1 = origin + np.array([dx, 0, 0])
    p2 = origin + np.array([0, dy, 0])
    return calibrate(arm, origin, p1, p2)


def draw_edge(arm, edge, dz=2, speed=100, wait=False):
    """
    1. Go above the first point in the edge.
    2. Lower the pen.
    3. Go through all of the points in the edge.
    4. Raise the pen above the last point.

    Arguments:
    edge -- an array of shape (3, number_of_points)
    dz -- the distance from the plane when the pen is lifted, in mm

    Returns the number of points drawn.
    """

    x = edge[0, 0]
    y = edge[1, 0]
    z = edge[2, 0]

    # Put pen in position, above the start point
    arm.set_position(
        x=x,
        y=y,
        z=z + dz,
        roll=180,
        pitch=0,
        yaw=0,
        speed=speed,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )

    # Draw
    for i in range(0, edge.shape[1]):
        x = edge[0, i]
        y = edge[1, i]
        z = edge[2, i]

        arm.set_position_aa(
            [x, y, z, 180, 0, 0],
            speed=speed,
            is_radian=0,
            wait=wait,
            radius=None,
            relative=False,
            mvacc=2000,
        )

    # Lift pen
    arm.set_position(
        x=0,
        y=0,
        z=dz,
        roll=0,
        pitch=0,
        yaw=0,
        speed=speed,
        is_radian=0,
        wait=True,
        radius=None,
        relative=True,
    )

    return len(edge)


def wait_for_input(password):
    while input(f'\nPlease enter "{password}" to restart : ') != password:
        continue


def draw_edges(arm, edges, dz=2, verbose=True, speed=100, pause_at_edge_nb=None, pause_after=None):
    """
    Draw the edges in order.

    Arguments:

    edges -- a list of point groups, each edge being of shape (3, number_of_points)

    dz -- the distance from the plane when the pen is lifted, in mm (default 2.0)

    verbose -- if True, print the progress as a percentage
    """

    already_paused = False

    if pause_after is not None:
        start = time()

    nb_points_drawn = 0
    nb_points = sum([len(edge) for edge in edges])

    for i, edge in enumerate(edges):
        nb_points_drawn += draw_edge(arm, edge, dz, speed)

        if verbose:
            print(f"{nb_points_drawn*100/nb_points}% complete...")

        if not already_paused:
            if pause_after is not None:
                current = time()
                duration = current - start
                if duration > pause_after:
                    already_paused = True
                    wait_for_input("continue")

            elif pause_at_edge_nb is not None and i > pause_at_edge_nb:
                already_paused = True
                wait_for_input("continue")
