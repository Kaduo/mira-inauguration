import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import findWall,absolute_coords,optimize_path, image_thresholding
import time
from numpy import linalg as LA
from photo2drawing import grouping_edges, plotting_contours



fig,ax = plt.subplots(1)
file = open('/Users/nicolas/Documents/Cours Dev IA/xArm-Python-SDK-master/mira_coords.pkl','rb')
mira_data = pickle.load(file)
xs=[]
ys=[]
for letter in mira_data:
    data = np.array(mira_data[letter])/500
    for x in data:
        xs.append(x[0])
        ys.append(x[1])
        
    ax.scatter(xs,ys,s=0.1)
    plt.show()