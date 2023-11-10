from utils import coords_converter,sort_groups
import numpy as np
from arms import get_big_drawing_arm, calibrate
import skimage as ski


from photo2drawing import grouping_edges

arm = get_big_drawing_arm()

above_origin = np.array([177, -118, 134])
above_p1 = np.array([550, -118, 137])
above_p2 = np.array([177,130, 137])

origin, p1, p2 = calibrate(arm, above_origin, above_p1, above_p2)

image = ski.io.imread("st jerome.jpg")
converter = coords_converter(list(reversed(image.shape[:2])), origin, p1, p2)

_, draw = grouping_edges(image, 1000, rescale=False)

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
  