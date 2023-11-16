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
    res = cv2.flip(res,1)
    res = res[res.shape[0] // 2 :, :]
    return res

def get_frame(cap):
    _, frame = cap.read()
    return process_frame(frame)

def show(image):
    logo = image
    # importing image on which we are going to  
    # apply watermark 
    img = cv2.imread("Data/cadre.png") 
    h_logo, w_logo, _ = logo.shape 

    # height and width of the image 
    h_img, w_img, _ = img.shape 
    center_y = int(h_img/2) 
    center_x = int(w_img/2) 
    
    # calculating from top, bottom, right and left 
    top_y = center_y - int(h_logo/2) -70
    bottom_y = top_y + h_logo 
    left_x = center_x - int(w_logo/2) 
    right_x = left_x + w_logo 
    destination = img[top_y:bottom_y, left_x:right_x] 
    result = cv2.addWeighted(destination,0, logo, 1, 0) 
    img[top_y:bottom_y, left_x:right_x] = result 
    cv2.imshow("MIRA", img)
    key = cv2.waitKey(1) & 0xFF
    return key
        
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
    key = show(im)

def photomaton_loop(cap, waiting_time=100):
    selected = False
    while not selected:
        frame = get_frame(cap)
        
        # height and width of the image 
        key = show(frame)
       
        
        if key == ord("c"):
            for i in range(waiting_time, 0, -1):
                im = cv2.putText(
                    frame,
                    f"{round(i/21)}",
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    5,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                key = show(im)
                time.sleep(0.01)
                frame = get_frame(cap)

            frame = get_frame(cap)
            key = show(frame)
            cv2.imwrite("image.jpg", frame)
            return frame
        if key == ord("e"):
            cv2.destroyWindow("MIRA")
            return


cadre = cv2.imread('data/cadre.png')
# cv2.imshow('cadre', cadre)
# cv2.waitKey(0)

# Close all windows
# cv2.destroyAllWindows()
h_cadre, w_cadre, _ = cadre.shape 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


with open("mira_correct.pkl", "rb") as f:
    mira_data = pickle.load(f)
    f.close()

arm = get_photomaton_arm()

above_origin = np.array([310, -60, 142])
above_p1 = np.array([448, -60, 142])
above_p2 = np.array([310, 50, 142])

while True:
    frame = photomaton_loop(cap, 70)

    edge_image = rgb2edge_image(frame)

    edge_image[15*edge_image.shape[0]//16:, 3*edge_image.shape[1]//4:] = True

    edges = edge_image2edges(edge_image, nb_edges = 1500)

    drawing_in_progress(edge_image)

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    origin, p1, p2 = calibrate(arm, [above_origin, above_p1, above_p2], relative_epsilon=0.25, absolute_epsilon=0)

    converter = CoordinatesConverter(list(reversed(edge_image.shape[:2])), origin, p1, p2)
    sorted_edges = sort_edges(edges)
    converted_edges = converter.convert_list_of_points(sorted_edges)

    draw_edges(arm, converted_edges)

    print("shape", edge_image.shape)
    for letter in mira_data:
        print("before", letter)
        letter *= edge_image.shape[0]
        letter /= 4
        letter[1] += 15*edge_image.shape[1]//16
        letter[0] += 3*edge_image.shape[0]//4
        letter = converter.convert(np.array(letter))
        draw_edge(arm, letter, wait=True, speed=50)
        print("after", letter)
    
    arm.set_position(x=0, y = 0, z=30, speed=100, wait=True, relative=True)