#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:05:41 2023

@author: nicolas
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np

import time
import skimage as ski


def find_surface(arm, x, y, z, Rx=180, Ry=0, Rz=0, absolute_epsilon=1, relative_epsilon=0, verbose=True):
    torques = []
    arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=Rx,
        pitch=Ry,
        yaw=Rz,
        speed=100,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )
    initial_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(
        x=x,
        y=y,
        z=z - 1,
        roll=Rx,
        pitch=Ry,
        yaw=Rz,
        speed=50,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )
    test1_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(
        x=x,
        y=y,
        z=z - 2,
        roll=Rx,
        pitch=Ry,
        yaw=Rz,
        speed=50,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )
    test2_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(
        x=x,
        y=y,
        z=z - 3,
        roll=Rx,
        pitch=Ry,
        yaw=Rz,
        speed=50,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )
    test3_torques = np.array(arm.joints_torque[0:6])
    mins = np.min([test1_torques, test2_torques, test3_torques, initial_torques], axis=0)
    maxs = np.max([test1_torques, test2_torques, test3_torques, initial_torques], axis=0)
    d = np.abs(maxs - mins)
    print(d)
    boundaries = [mins - relative_epsilon*d - absolute_epsilon, maxs + relative_epsilon*d + absolute_epsilon]
    print(boundaries)
    actual_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(
        x=x,
        y=y,
        z=z - 100,
        roll=Rx,
        pitch=Ry,
        yaw=Rz,
        speed=1,
        is_radian=0,
        wait=False,
        radius=None,
        relative=False,
    )
    while (boundaries[0] < actual_torques).all() and (actual_torques < boundaries[1]).all():
        actual_torques = np.array(arm.joints_torque[0:6])
        pos = arm.position_aa
        torques.append(actual_torques)
        if verbose:
            print("i'm waiting for collision")
    if verbose:
        print("I collided")
    arm.set_state(4)
    time.sleep(1)
    arm.set_state(0)
    arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=Rx,
        pitch=Ry,
        yaw=Rz,
        speed=100,
        is_radian=0,
        wait=True,
        radius=None,
        relative=False,
    )
    return pos, torques



class absolute_coords:
    def __init__(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.x1, self.y1, self.z1 = x1, y1, z1
        self.x2, self.y2, self.z2 = x2, y2, z2
        self.mat = np.array(
            [
                [self.x1 - self.x0, self.x2 - self.x0],
                [self.y1 - self.y0, self.y2 - self.y0],
                [self.z1 - self.z0, self.z2 - self.z0],
            ]
        )

    def convert(self, points):
        return np.dot(self.mat, points) + np.array([[self.x0], [self.y0], [self.z0]])


def optimize_path(input_data, threshold):
    data = input_data / np.max(input_data)
    plt.figure()

    for seg in data:
        plt.plot(seg.T[0], seg.T[1])
        plt.gca().invert_yaxis()

    data = list(data)
    new_data = [[data[0]]]
    id_group = 0
    id_ = 0
    for id_line in range(len(data)):
        if len(data) > 1:
            dmin = LA.norm(data[0][1] - data[1][0])
            id_min = 0
            for id_seg in range(len(data)):
                d = LA.norm(new_data[id_group][id_][1] - data[id_seg][0])
                if d <= dmin:
                    id_min = id_seg
                    dmin = d
        else:
            id_min = 0
        id_ += 1
        if dmin > threshold:
            id_group += 1
            id_ = 0
            new_data.append([])
        new_data[id_group].append(data[id_min])
        data.pop(id_min)

    compression = len(new_data) / len(input_data)
    return new_data, compression


def sort_edges(edges):
    """Sort edges so as to minimize useless movements.
    
    Argument:

    edges -- a list of edges, each of shape [nb_of_points, 2]
    """
    edges = edges.copy()
    sorted_edges = [edges.pop(0)]
    for _ in range(len(edges) - 1):

        def dist_to_last_edge(other_edge):
            return min(
                LA.norm(sorted_edges[-1][-1] - other_edge[0]),
                LA.norm(sorted_edges[-1][-1] - other_edge[-1]),
            )

        def better_flipped(other_edge):
            return LA.norm(sorted_edges[-1][-1] - other_edge[-1]) < LA.norm(
                sorted_edges[-1][-1] - other_edge[0]
            )

        edges = sorted(edges, key=dist_to_last_edge)

        edge_to_add = edges.pop(0)
        if better_flipped(edge_to_add):
            edge_to_add = edge_to_add[::-1]

        sorted_edges.append(edge_to_add)

    return sorted_edges


def image_thresholding(image):
    image = ski.restoration.denoise_bilateral(
        image, sigma_color=0.5, sigma_spatial=2, channel_axis=-1
    )
    image = ski.exposure.equalize_adapthist(image)
    image = ski.color.rgb2gray(image)
    hist = plt.hist(image.ravel(), bins=256)
    histo = np.array(hist[0])
    maximum = histo.argmax() / 256
    thresholds = ski.filters.threshold_multiotsu(image)
    edge_image = ski.feature.canny(image, low_threshold=0.1, high_threshold=0.2)
    hough_lines = ski.transform.probabilistic_hough_line(
        edge_image, line_length=6, line_gap=2, threshold=20
    )
    hough_lines = np.array(hough_lines)
    return edge_image, hough_lines
