from xarm.wrapper import XArmAPI
from utils import findWall
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

    calibrated_origin, _ = findWall(arm, origin[0], origin[1], origin[2], Rx, Ry, Rz)[:3].reshape((1,-1))
    calibrated_p1, _ = findWall(arm, p1[0], p1[1], p1[2], Rx, Ry, Rz)[:3].reshape((1,-1))
    calibrated_p2, _ = findWall(arm, p2[0], p2[1], p2[2], Rx, Ry, Rz)[:3].reshape((1,-1))

    return calibrated_origin, calibrated_p1, calibrated_p2

def calibrate_from_dimensions(arm, origin, dx, dy):
    p1 = origin + np.array([dx, 0, 0])
    p2 = origin + np.array([0, dy, 0])
    return calibrate(origin, p1, p2)