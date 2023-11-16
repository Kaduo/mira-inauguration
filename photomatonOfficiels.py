#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:45:49 2023

@author: nicolas
"""

import pandas as pd
import cv2

import numpy as np

import matplotlib.pyplot as plt

import time
import pickle


from xarm.wrapper import XArmAPI
from utils import find_surface, absolute_coords, optimize_path, image_thresholding, sort_edges
from numpy import linalg as LA
from photo2drawing import grouping_edges, plot_edges, rgb2edges, rgb2edge_image, edge_image2edges
from arms import get_photomaton_arm, calibrate, draw_edges, draw_edge
from coordinates import CoordinatesConverter


import skimage as ski

def process_frame(frame):
    res = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res = res[res.shape[0] // 2 :, :]
    return res

def get_frame(cap):
    _, frame = cap.read()
    return process_frame(frame)

def drawing_in_progress(edge_image):
    edge_image = np.array(edge_image, dtype=np.uint8) * 255
    edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(edge_image, 0.5, frame, 0.5, 0)

    im = cv2.putText(
        superimposed,
        "dessin en cours...",
        (int(frame.shape[1] / 2) - 300, int(frame.shape[0]) - 100),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        3,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("MIRA", im)
    cv2.waitKey(1) & 0xFF

def photomaton_loop(cap, waiting_time=100):
    selected = False
    while not selected:
        frame = get_frame(cap)
        im = cv2.putText(
            frame,
            "Portrait robot !",
            (300, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("MIRA", im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            for i in range(waiting_time, 0, -1):
                im = cv2.putText(
                    frame,
                    f"{round(i/30)}",
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("MIRA", im)
                key = cv2.waitKey(1) & 0xFF
                time.sleep(0.01)
                frame = get_frame(cap)

            frame = get_frame(cap)
            cv2.imshow("MIRA", frame)
            key = cv2.waitKey(1) & 0xFF
            cv2.imwrite("image.jpg", frame)
            return frame
        if key == ord("e"):
            cv2.destroyWindow("MIRA")
            return



cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


with open("mira_correct.pkl", "rb") as f:
    mira_data = pickle.load(f)
    f.close()

arm = get_photomaton_arm()

above_origin = np.array([420, -80, 194])
above_p1 = np.array([560, -80, 194])
above_p2 = np.array([420, 50, 194])

frame = ski.io.imread('data/patxi.jpg')#photomaton_loop(cap, 0)

edge_image = rgb2edge_image(frame)

edge_image[15*edge_image.shape[0]//16:, 3*edge_image.shape[1]//4:] = True

edges = edge_image2edges(edge_image, nb_edges = 700)

drawing_in_progress(edge_image)

arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

origin, p1, p2 = calibrate(arm, [above_origin, above_p1, above_p2], absolute_epsilon=[2,2,3])
converter = CoordinatesConverter(list(reversed(edge_image.shape[:2])), origin, p1, p2)
sorted_edges = sort_edges(edges)
converted_edges = converter.convert_list_of_points(sorted_edges)


shift_mira = np.array([0,10,0])
origin_mira = origin+(3/4*(p2-origin)+15/16*(p1-origin)) - shift_mira
p1_mira = origin_mira+1/16*(p1-origin)
p2_mira = origin_mira+1/4*(p2-origin)
mira_image = [np.max(mira_data[1]), np.max(mira_data[0])]
converterMIRA = CoordinatesConverter(mira_image, origin_mira, p1_mira, p2_mira)

draw_edges(arm, converted_edges)

for letter in mira_data:
    
    points = converterMIRA.convert(letter)
    draw_edge(arm, points, speed=50, wait=True)


print("shape", edge_image.shape)
# for letter in mira_data:
    # print("before", letter)
    # letter *= edge_image.shape[0]
    # letter /= 4
    # letter[1] += 15*edge_image.shape[1]//16
    # letter[0] += 3*edge_image.shape[0]//4
    # letter = converter.convert(np.array(letter))
    # draw_edge(arm, letter, wait=True, speed=50)
    # print("after", letter)

arm.set_position(x=0, y = 0, z=30, speed=100, wait=True, relative=True)
