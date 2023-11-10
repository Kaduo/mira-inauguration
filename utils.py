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


import os
import sys
import time
import skimage as ski

def findWall(arm,x0,y0,z0,Rx0,Ry0,Rz0):
    torques = []
    arm.set_position(x=x0, y=y0, z=z0, roll=Rx0, pitch=Ry0, yaw=Rz0, speed=100, is_radian=0, wait=True,radius = None, relative = False)
    initial_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(x=x0, y=y0, z=z0-1, roll=Rx0, pitch=Ry0, yaw=Rz0, speed=50, is_radian=0, wait=True,radius = None, relative = False)
    test1_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(x=x0, y=y0, z=z0-2, roll=Rx0, pitch=Ry0, yaw=Rz0, speed=50, is_radian=0, wait=True,radius = None, relative = False)
    test2_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(x=x0, y=y0, z=z0-3, roll=Rx0, pitch=Ry0, yaw=Rz0, speed=50, is_radian=0, wait=True,radius = None, relative = False)
    test3_torques = np.array(arm.joints_torque[0:6])
    epsilon = 1
    mins = np.min([test1_torques,test2_torques,test3_torques, initial_torques],axis=0)
    maxs = np.max([test1_torques,test2_torques,test3_torques, initial_torques],axis=0)
    d = np.abs(maxs - mins)
    print(d)
    boundaries = [mins-epsilon, maxs + epsilon]
    print(boundaries)
    actual_torques = np.array(arm.joints_torque[0:6])
    arm.set_position(x=x0, y=y0, z=z0-5, roll=Rx0, pitch=Ry0, yaw=Rz0, speed=1, is_radian=0, wait=False,radius = None, relative = False)
    #print(norm_evolution)
    while (boundaries[0]<actual_torques).all() and (actual_torques<boundaries[1]).all():
        actual_torques = np.array(arm.joints_torque[0:6])
        pos=arm.position_aa
        torques.append(actual_torques)
        print("i'm waiting for collision")
    #arm.motion_enable(enable=False)
    print("I collided")
    arm.set_state(4)
    time.sleep(1)
    arm.set_state(0)
    arm.set_position(x=x0, y=y0, z=z0, roll=Rx0, pitch=Ry0, yaw=Rz0, speed=100, is_radian=0, wait=True,radius = None, relative = False)
    return pos,torques


class absolute_coords:
    def __init__(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        self.x0, self.y0, self.z0  = x0, y0, z0
        self.x1, self.y1, self.z1  = x1, y1, z1
        self.x2, self.y2, self.z2  = x2, y2, z2
        self.mat = np.array([[self.x1 - self.x0, self.x2 - self.x0],
                            [self.y1 - self.y0, self.y2 - self.y0],
                            [self.z1 - self.z0, self.z2 - self.z0]])
    def convert(self, points):
        return np.dot(self.mat, points) + np.array([[self.x0],
                                                    [self.y0],
                                                    [self.z0]])
    
    

class coords_converter:
    def __init__(self, image_shape, origin, point1, point2):
        image_length = max(image_shape)
        image_length_idx = np.argmax(image_shape)
        image_width = min(image_shape)
        image_width_idx = np.argmin(image_shape)
        e1 = point1 - origin
        e2 = point2 - origin
        l1 = LA.norm(e1)
        l2 = LA.norm(e2)

        plane_length = max(l1, l2)
        plane_length_idx = np.argmax([l1, l2])
        plane_length_vec = [e1, e2][plane_length_idx]
        plane_width = min(l1, l2)
        plane_width_idx = np.argmin([l1, l2])
        plane_width_vec = [e1, e2][plane_width_idx]

        image_ratio = image_length/image_width
        plane_ratio = plane_length/plane_width

        mat = np.zeros((2, 3))

        if image_ratio > plane_ratio:
            mat[image_length_idx] = plane_length_vec/image_length
            mat[image_width_idx] = (plane_width_vec*plane_length)/(plane_width*image_length)
        else:
            mat[image_width_idx] = plane_width_vec/image_width
            mat[image_length_idx] = (plane_length_vec*plane_width)/(plane_length*image_width)
        
        self.mat = mat.T
        self.origin = origin.T

    def convert(self, points):
        return np.dot(self.mat, points) + self.origin


def optimize_path(input_data,threshold):
    data = input_data/np.max(input_data)
    plt.figure()

    for seg in data:
        plt.plot(seg.T[0],seg.T[1])
        plt.gca().invert_yaxis()

    data=list(data)
    new_data=[[data[0]]]
    id_group = 0
    id_=0
    for id_line in range(len(data)):
        if len(data)>1:
            dmin=LA.norm(data[0][1]-data[1][0])
            id_min=0
            for id_seg in range(len(data)):
                d=LA.norm(new_data[id_group][id_][1]-data[id_seg][0])
                if d<=dmin:
                    id_min = id_seg
                    dmin=d
        else:
            id_min=0
        id_+=1
        if dmin>threshold:
            id_group+=1
            id_=0
            new_data.append([])
        new_data[id_group].append(data[id_min])
        data.pop(id_min)
        # print(id_line)
        
    compression = len(new_data)/len(input_data)
    return new_data,compression

def sort_line_groups(groups):
    groups = groups.copy()
    new_groups = [groups.pop(0)]
    for _ in range(len(groups) - 1):

        def dist_to_last_group(other_group):
            return min(LA.norm(new_groups[-1][-1] - other_group[0]),
                        LA.norm(new_groups[-1][-1] - other_group[-1]))

        def better_flipped(other_group):
            return LA.norm(new_groups[-1][-1] - other_group[-1]) < LA.norm(new_groups[-1][-1] - other_group[0])

        groups = sorted(groups, key=dist_to_last_group)
        
        group_to_add = groups.pop(0)
        if better_flipped(group_to_add):
            group_to_add = group_to_add[::-1]

        new_groups.append(group_to_add)
        
    return new_groups




def image_thresholding(image):
    image = ski.restoration.denoise_bilateral(image, sigma_color=0.5, sigma_spatial=2, channel_axis=-1)
    image = ski.exposure.equalize_adapthist(image)
    image = ski.color.rgb2gray(image)
    # image = ski.transform.resize(image, [1000,1000],preserve_range=True)
    hist = plt.hist(image.ravel(),bins=256)
    histo = np.array(hist[0])
    maximum=histo.argmax()/256
    thresholds = ski.filters.threshold_multiotsu(image)
    edge_image = ski.feature.canny(image, low_threshold = 0.1, high_threshold=0.2)
    hough_lines = ski.transform.probabilistic_hough_line(edge_image, line_length=6, line_gap=2, threshold=20)
    hough_lines = np.array(hough_lines)
    return edge_image,hough_lines


    

# image = ski.io.imread("Nico_1.jpg")

# canny,lines = image_thresholding(image)
# arm = XArmAPI(ip, is_radian=True)
# arm.motion_enable(enable=True)
# arm.set_mode(0)
# arm.set_state(state=0)
# #arm.set_collision_sensitivity(3)


# x0 = 280
# y0 = 190
# z0 = 160
# Rx0 = 180
# Ry0 = 0
# Rz0 = 0

# point3,torques3 = findWall(arm,x0,y0,z0,Rx0,Ry0,Rz0)
# plt.figure(3)
# plt.plot(torques3)