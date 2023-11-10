import sys
import os
from xarm.wrapper import XArmAPI

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