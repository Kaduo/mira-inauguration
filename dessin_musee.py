from xarm.wrapper import XArmAPI
from utils import findWall,absolute_coords,optimize_path,image_thresholding,coords_converter,sort_groups
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from photo2drawing import grouping_edges, plotting_contours


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = '192.168.1.213'
        if not ip:
            print('input error, exit')
            sys.exit(1)


arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)


# Origin

x0 = 200
y0 = -120
z0 = 140
Rx0 = 180
Ry0 = 0
Rz0 = 0

point2,torques2 = findWall(arm,x0,y0,z0,Rx0,Ry0,Rz0)
plt.figure(2)
plt.plot(torques2)

x0 = 560
y0 = -120
z0 = 140

point3,torques3 = findWall(arm,x0,y0,z0,Rx0,Ry0,Rz0)
plt.figure(1)
plt.plot(torques3)


x0 = 200
y0 = 148
z0 = 140

torques = []

point1,torques1 = findWall(arm,x0,y0,z0,Rx0,Ry0,Rz0)
plt.figure(3)
plt.plot(torques1)

print(point1)
print(point2)
print(point3)

point1 = np.array(point1[:3]).reshape((1,-1))
point2 = np.array(point2[:3]).reshape((1,-1))
point3 = np.array(point3[:3]).reshape((1,-1))


# first_line = converter.convert(np.array([[0,0], [200, 400]]).T)
# last_line = converter.convert(np.array([[200,0], [0, 400]]).T)

# print(first_line)
# print(last_line)

# for l in [first_line, last_line]:
#     p1 = l[:,0]
#     p2 = l[:,1]
#     arm.set_position_aa([p1[0], p1[1], 140, 180, 0, 0], speed=100, is_radian=0, wait=True, radius = None, relative = False)
#     arm.set_position_aa([p1[0], p1[1], p1[2], 180, 0, 0], speed=100, is_radian=0, wait=True, radius = None, relative = False)
#     arm.set_position_aa([p2[0], p2[1], p2[2], 180, 0, 0], speed=100, is_radian=0, wait=True, radius = None, relative = False)
#     arm.set_position_aa([p2[0], p2[1], 140, 180, 0, 0], speed=100, is_radian=0, wait=True, radius = None, relative = False)

image = ski.io.imread("st jerome.jpg")
converter = coords_converter(list(reversed(image.shape[:2])), point2, point1, point3)

print(image.shape)

_,draw = grouping_edges(image, 1000, rescale=False)

draw = sort_groups(draw)

idx = 0

for group in draw:
    
    new_points = converter.convert(group.T)
    
    percentage = idx/len(draw)
    print(int(percentage*100))
    
    x = new_points[0, 0]
    y = new_points[1, 0]
    z = new_points[2, 0]
    
    arm.set_position(x=x, y=y, z=z+2, roll=180, pitch=0, yaw=0, speed=100, is_radian=0, wait=True,radius = None, relative = False)
        
    for i in range(0,new_points.shape[1]):
        
        x = new_points[0, i]
        y = new_points[1, i]
        z = new_points[2, i]
        arm.set_position_aa([x, y, z, 180, 0, 0], speed=100, is_radian=0, wait=False,radius = None, relative = False,mvacc=2000)
        
    idx+=1
    arm.set_position(x=0, y=0, z=2, roll=0, pitch=0, yaw=0, speed=100, is_radian=0, wait=True,radius = None, relative = True)
  