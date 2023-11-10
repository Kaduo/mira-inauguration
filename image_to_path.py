#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:20:30 2023

@author: nicolas
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import feature

# image = cv2.imread('logo-MIRA.png')
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def path_generation(image):
    grayscale_image = image
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate the magnitude of the gradient to find edges
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Convert the gradient magnitude to an 8-bit image
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    # Threshold the image to create a binary edge mask
    _, edge_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    binary = 255-cv2.threshold(grayscale_image,150,255,cv2.THRESH_BINARY)[1]
    #fix,ax=plt.subplots(2)
    #ax[0].imshow(edge_mask)
    #ax[1].imshow(binary)
    
    # Find contours using the findContours function
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = image.copy()*0
    # Draw the contours on a copy of the original image
    contour_image = cv2.drawContours(blank_image, contours, -1, (255, 255, 255), 2)
    #plt.imshow(contour_image)
    image_with_points=blank_image
    for segments in contours:
        for point in segments:
            cv2.circle(image_with_points, point[0], 5, (255, 0, 0), -1)
    #plt.imshow(image_with_points,cmap='gray')
    
    #Here we have to look for the vector direction of our points : 
    
    directions = []
    
    for segments in contours :
        for idx_point in range(len(segments)-1) :
            directions.append(segments[idx_point+1][0]-segments[idx_point][0])
    #plt.plot(directions)
    
    from scipy.interpolate import make_interp_spline
    
    plt.figure()
    drawings=[]
    for contour in contours :
        
        
        spline = make_interp_spline(np.linspace(0, 1, contour.shape[0]), contour[:, 0])
        
        ts = np.linspace(0, 1, contour.shape[0]*3)
        
        
        
        x_new, y_new = spline(ts).T
        x_new=np.append(x_new,x_new[0])
        y_new=np.append(y_new,y_new[0])
        drawings.append([x_new,y_new])
        plt.gca().invert_yaxis()
        #plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'o')
        plt.plot(x_new, y_new, '-')
        
    plt.show()
    maximum=0
    for drawing in drawings:
        if maximum<np.max(drawing):
            maximum = np.max(drawing)
    for id_drawing in range(len(drawings)):
        drawings[id_drawing]/=maximum
        
    return drawings



def contour_maker(image):
    grayscale_image = image
    # sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=5)
    # sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=5)
    
    # # Calculate the magnitude of the gradient to find edges
    # gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # # Convert the gradient magnitude to an 8-bit image
    # gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    # # Threshold the image to create a binary edge mask
    # _, edge_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    binary = 255-cv2.threshold(grayscale_image,150,255,cv2.THRESH_BINARY)[1]

    
    # Find contours using the findContours function
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = image.copy()*0
    # Draw the contours on a copy of the original image
    contour_image = cv2.drawContours(blank_image, contours, -1, (255, 255, 255), 2)
    #plt.imshow(contour_image)
    image_with_points=blank_image

    return contours



def corner_maker(image):
    binary = 255-cv2.threshold(image,150,255,cv2.THRESH_BINARY)[1]
    harris_image = feature.corner_harris(binary)
    coords = feature.corner_peaks(harris_image, min_distance=1, threshold_rel=0.001)
    return coords

# binary, coords = corner_maker(image)
# plt.figure()
# plt.imshow(binary,cmap='gray')
# plt.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',linestyle='None', markersize=6)

# contours = contour_maker(image)
# plt.figure()
# for contour in contours:
#     plt.plot(contour[:,:,0],contour[:,:,1])
# plt.gca().invert_yaxis()
    

# def make_segments(contours):
#     direction = []
#     for contour in contours:
#         used_contour = contour.reshape((len(contour),2))
#         vect=contour[4]-contour[0]
#         a,b=vect[0]
#         first_direction = np.arctan(b/a)
#         for i in range(len(contour)-2):
#             vect=contour[i+2]-contour[i]
#             a,b=vect[0]
#             direction.append(np.arctan(b/a))
#         segments=[]
#         segment=[]
#         in_seg=0
        
#         for id_dir in range(1,len(direction)):
#             prev_dir=direction[id_dir-1]
#             actual_dir = direction[id_dir]
#             if np.abs(actual_dir-prev_dir)<0.3:
#                 in_seg=1
#                 segment.append(contour[id_dir][0])
#             else:
#                 in_seg=0
#                 segments.append(segment)
#                 segment=[]
#         return direction,segments
            
    
    
# direction,segments = make_segments(contours)
# plt.figure()
# plt.plot(direction)
# plt.figure()
# for seg in segments:
#     if len(seg)>1:
#         pt1 = seg[0]
#         pt2 = seg[-1]
#         plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]])
    
    
