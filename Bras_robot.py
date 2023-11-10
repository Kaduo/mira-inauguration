#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:32:28 2023

@author: nicolas
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#######################################################
"""
Just for test example
"""
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = '192.168.1.207'
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################


arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Position initiale du robot
x0 = 200
z0 = 140
y0 = 0
arm.set_position(x=x0, y=y0, z=z0, roll=180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)

# Module de gestion et recuperation des images
debug = False
debug_path = 'poslog.debug'
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray

# For webcam input:
cap1 = cv2.VideoCapture(0) # image webcam
cap2 = cv2.VideoCapture(1) # image caméra

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap1.isOpened():
        if cap2.isOpened():
            success1, image1 = cap1.read()
            success2, image2 = cap2.read()       
            # création de la Box d'environnement
            x_rect = int(640*0.25)
            y_rect = int(480*0.9)
            x_rect2 = int(640*0.90)
            y_rect2 = int(480*0.1)
            cv2.rectangle(image1,(x_rect,y_rect),(x_rect2,y_rect2),(0,0,255),2)
            cv2.rectangle(image2,(x_rect,y_rect),(x_rect2,y_rect2),(0,0,255),2)
            if not success1:
                print("Ignoring empty camera1 frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
            if not success2:
                print("Ignoring empty camera2 frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image1.flags.writeable = False
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            results1 = hands.process(image1)
            image2.flags.writeable = False
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            results2 = hands.process(image2)
            if debug == True:
                # print(results.pose_landmarks) TODO: Mettre les resultats dans un fichier
                pass
            if results1.multi_hand_landmarks and results2.multi_hand_landmarks: # Renvoie les cordonnées de l'index droit transformer en pourcentage
                posX = int(100*results1.multi_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
                posY = int(100*results1.multi_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
                posZ = int(100*results2.multi_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z)
            else:
                posX = 0
                posY = 0
                posZ = 0
                    
            # Draw the pose annotation on the image.
            image1.flags.writeable = True
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            image2.flags.writeable = True
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image1,
                results1.hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image2,
                results2.hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            image1=cv2.flip(image1, 1) # Inverse l'image
            # Affichage text sur l'image
            cv2.putText(image1,f"x:{posX}",[80,80], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.putText(image1,f"y:{posY}",[80,120], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.putText(image2,f"z:{posZ}",[80,80], cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            hStack = np.hstack((image1,image2))
            cv2.imshow('hStack', hStack) # Affichage l'image avec les modifications
            cv2.waitKey(5)
        #transform MP for robot coord
        
            '''
            Module d'initialisation de la position de départ du pilote dans l'espace
            '''
            def z_Humain(posZ):
                zR = ( 13 * posZ ) -510
                return zR
            
            xR = z_Humain(posX)
            yR = (posX * 8) - 460
            zR = (posY * -6.5) + 595
                
            #clip to the rail
            xR=np.clip(xR,x0+10,530)
            yR=np.clip(yR,y0-260,y0+260)
            zR=np.clip(zR,z0+10,z0+530)
            # if debug == True:
            #     print(xR,yR,zR)
            arm.set_position(x=xR, y=yR, z=zR, roll=180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
    
