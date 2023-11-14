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
from utils import find_surface, absolute_coords, optimize_path, image_thresholding
from numpy import linalg as LA
from photo2drawing import grouping_edges, plot_edges, rgb2edges
from arms import get_photomaton_arm, calibrate
from coordinates import CoordinatesConverter


import skimage as ski



def process_frame(frame):
    res = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res = res[res.shape[0] // 2 :, :]
    return res

def get_frame(cap):
    ret, frame = cap.read()
    return process_frame(frame)

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


number_of_lines = 300

file = open("mira_coords.pkl", "rb")
mira_data = pickle.load(file)

arm = get_photomaton_arm()

above_origin = np.array([260, -63, 193])
above_p1 = np.array([260, 63, 193])
above_p2 = np.array([415, -63, 193])

while True:
    frame = photomaton_loop(cap, 0)

    edges, edge_image = rgb2edges(frame, return_edge_image=True)
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

    # print(image_edge.shape)
    cv2.imshow("MIRA", im)
    key = cv2.waitKey(1) & 0xFF

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    origin, p1, p2 = calibrate(arm, above_origin, above_p1, above_p2, epsilon=1)

    converter = CoordinatesConverter(frame.shape[:2], origin, p1, p2)
    idx = 0

    for group in edges:
        new_points = converter.convert(group.T)
        percentage = idx / len(edges)
        print(int(percentage * 100))

        x = new_points[0, 0]
        y = new_points[1, 0]
        z = new_points[2, 0]

        arm.set_position(
            x=x,
            y=y,
            z=z + 5,
            roll=180,
            pitch=0,
            yaw=0,
            speed=100,
            is_radian=0,
            wait=True,
            radius=None,
            relative=False,
        )

        for i in range(0, new_points.shape[1]):
            x = new_points[0, i]
            y = new_points[1, i]
            z = new_points[2, i]
            arm.set_position_aa(
                [x, y, z, 180, 0, 0],
                speed=100,
                is_radian=0,
                wait=False,
                radius=None,
                relative=False,
                mvacc=2000,
            )

        idx += 1
        arm.set_position(
            x=0,
            y=0,
            z=5,
            roll=0,
            pitch=0,
            yaw=0,
            speed=100,
            is_radian=0,
            wait=True,
            radius=None,
            relative=True,
        )

    for letter in mira_data:
        data = np.array(mira_data[letter])
        data = np.array([data.T[1], data.T[0]])

        ma = None
        mi = None
        for letter in mira_data.values():
            maybe_ma = np.max(np.array(letter))
            maybe_mi = np.min(np.array(letter))
            if ma is None or maybe_ma > ma:
                ma = maybe_ma
            if mi is None or maybe_mi < mi:
                mi = maybe_mi
        data = (data - mi) / (ma - mi)
        print(data)
        data /= 4
        data[1] += frame.shape[0] / max(frame.shape[0], frame.shape[1]) + 0.04
        # data[1] -= 0.3
        data[0] += 1 - 1 / 4
        new_points = converter.convert(data)
        x = new_points[0, 0]
        y = new_points[1, 0]
        z = new_points[2, 0]
        arm.set_position(
            x=x,
            y=y,
            z=z + 2,
            roll=180,
            pitch=0,
            yaw=0,
            speed=100,
            is_radian=0,
            wait=True,
            radius=None,
            relative=False,
        )

        for i in range(0, new_points.shape[1] + 1):
            idx = i % new_points.shape[1]
            x = new_points[0, idx]
            y = new_points[1, idx]
            z = new_points[2, idx]
            arm.set_position_aa(
                [x, y, z, 180, 0, 0],
                speed=10,
                is_radian=0,
                wait=True,
                radius=None,
                relative=False,
            )

        idx += 1
        arm.set_position(
            x=0,
            y=0,
            z=2,
            roll=0,
            pitch=0,
            yaw=0,
            speed=25,
            is_radian=0,
            wait=True,
            radius=None,
            relative=True,
        )

    arm.set_position(
        x=250,
        y=0,
        z=z + 20,
        roll=180,
        pitch=0,
        yaw=0,
        speed=50,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )

    selected = False
# R_mira = np.array([x_new,y_new])
# new_points = abs_coords.convert(R_mira)
