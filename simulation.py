#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:57:37 2023

@author: nicolas
"""
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import findWall,absolute_coords,optimize_path, image_thresholding
import time
from numpy import linalg as LA
from photo2drawing import grouping_edges, plotting_contours



idx=0
x2 = 170
y2 = 70
z2 = 235

x0 = 310
y0 = 70
z0 = 235

x1 = 310
y1 = -70
z1 = 235

abs_coords = absolute_coords(x0, y0, z0, x1, y1, z1, x2, y2, z2)
file = open('/Users/nicolas/Documents/Cours Dev IA/xArm-Python-SDK-master/mira_coords.pkl','rb')
mira_data = pickle.load(file)

import skimage as ski

selected =0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    while selected == 0:
        
        ret, frame = cap.read()
        im = cv2.putText(frame, 'Portrait robot !', (300,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('MIRA', im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            
            for i in range(10,0,-1):
                im = cv2.putText(frame, f'{round(i/30)}', (int(frame.shape[1]/2),int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('MIRA', im)
                key = cv2.waitKey(1) & 0xFF
                time.sleep(0.01)
                ret, frame = cap.read()
            cv2.imshow('MIRA', frame)
            key = cv2.waitKey(1) & 0xFF
            cv2.imwrite('image.jpg',frame)
            selected=1
        if key == ord('e'):
            cv2.destroyWindow('MIRA')
            break
    image = ski.io.imread('image.jpg')

    edge,draw = grouping_edges(image, 1000)
    edge = np.array(edge,dtype=np.uint8)*255
    image_edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(image_edge,0.5,image,0.5,0)
    im = cv2.putText(superimposed, 'dessin en cours...', (int(frame.shape[1]/2)-300,int(frame.shape[0])-100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)

    #print(image_edge.shape)
    cv2.imshow('MIRA', im)
    key = cv2.waitKey(1) & 0xFF
    time.sleep(10)
    #fig,ax = plt.subplots(1)
    xs=[]
    ys=[]
    c=[]
    for group in draw:
        

            
        new_points = abs_coords.convert(group.T)
        
        
        percentage = idx/len(draw)
        print(percentage)
        
    
            
            #plot here
            
        
        
            # ax.ylim([-20,120])
            
        for i in range(0,new_points.shape[1]):
            
            x = new_points[0, i]
            y = new_points[1, i]
            z = new_points[2, i]
            
            xs.append(x)
            ys.append(y)
            c.append(idx)
        idx=idx+1
    
    for letter in mira_data:
        data = np.array(mira_data[letter])/1200
        new_points = abs_coords.convert(group.T)
        for i in range(0,new_points.shape[1]):
            
            x = new_points[0, i]
            y = new_points[1, i]
            z = new_points[2, i]
            
    #ax.scatter(xs,ys,c=c,s=0.1)
    #plt.show()
    selected = 0

    
    
    
