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
from photo2drawing import grouping_edges, plot_edges
from arms import get_photomaton_arm


import skimage as ski


number_of_lines = 700


file = open("mira_coords.pkl", "rb")
mira_data = pickle.load(file)

arm = get_photomaton_arm()


def photomaton_loop(cap, waiting_time=100):
    selected = False
    while not selected:
        _, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = frame[frame.shape[0] // 2 :, :].copy()
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
                _, frame = cap.read()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = frame[frame.shape[0] // 2 :, :].copy()

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

selected = False

while True:
    while not selected:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = frame[frame.shape[0] // 2 :, :].copy()
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
            for i in range(100, 0, -1):
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
                ret, frame = cap.read()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = frame[frame.shape[0] // 2 :, :].copy()

            cv2.imshow("MIRA", frame)
            key = cv2.waitKey(1) & 0xFF
            cv2.imwrite("image.jpg", frame)
            selected = True
        if key == ord("e"):
            cv2.destroyWindow("MIRA")
            break
    image = ski.io.imread("image.jpg")

    edge, draw = grouping_edges(image, number_of_lines)
    edge = np.array(edge, dtype=np.uint8) * 255
    image_edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(image_edge, 0.5, image, 0.5, 0)
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

    x0 = 260
    y0 = -63
    z0 = 193
    Rx0 = 180
    Ry0 = 0
    Rz0 = 0

    point2, torques2 = find_surface(arm, x0, y0, z0, Rx0, Ry0, Rz0)
    plt.figure(2)
    plt.plot(torques2)

    x0 = 260
    y0 = 63

    point3, torques3 = find_surface(arm, x0, y0, z0, Rx0, Ry0, Rz0)
    plt.figure(1)
    plt.plot(torques3)

    x0 = 415
    y0 = -63

    torques = []

    point1, torques1 = find_surface(arm, x0, y0, z0, Rx0, Ry0, Rz0)
    plt.figure(3)
    plt.plot(torques1)

    x0, y0, z0 = point2[0], point2[1], point2[2]
    x1, y1, z1 = point3[0], point3[1], point3[2]
    x2, y2, z2 = point1[0], point1[1], point1[2]

    print(point1)
    print(point2)
    print(point3)

    abs_coords = absolute_coords(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    idx = 0

    for group in draw:
        new_points = abs_coords.convert(group.T)

        percentage = idx / len(draw)
        print(int(percentage * 100))

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
            z=2,
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
        data[1] += image.shape[0] / max(image.shape[0], image.shape[1]) + 0.04
        # data[1] -= 0.3
        data[0] += 1 - 1 / 4
        new_points = abs_coords.convert(data)
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
