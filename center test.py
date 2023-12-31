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
from arms import get_photomaton_arm, calibrate, draw_edges
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


file = open("mira_coords.pkl", "rb")
mira_data = pickle.load(file)

arm = get_photomaton_arm()

above_origin = np.array([260, -63, 195])
above_p1 = np.array([415, -63, 195])
above_p2 = np.array([260, 63, 195])

edges = [np.array([[0,0], [200,210]]), np.array([[200,0], [0,210]])]

origin, p1, p2 = calibrate(arm, above_origin, above_p1, above_p2, epsilon=0.5)

converter = CoordinatesConverter([200,210], origin, p1, p2)
sorted_edges = sort_edges(edges)
converted_edges = converter.convert_list_of_points(edges)

idx = 0


draw_edges(arm, converted_edges)