from coordinates import CoordinatesConverter
from utils import sort_edges
import numpy as np
from arms import get_big_drawing_arm, calibrate, draw_edges
import skimage as ski

from photo2drawing import rgb2edges

arm = get_big_drawing_arm()

above_origin = np.array([200, -120, 138])
above_p1 = np.array([560, -120, 140])
above_p2 = np.array([200, 138, 139])

origin, p1, p2 = calibrate(arm, above_origin, above_p1, above_p2)

image = ski.io.imread("data/st jerome.jpg")
converter = CoordinatesConverter(list(reversed(image.shape[:2])), origin, p1, p2)

edges = rgb2edges(image, nb_edges=1000)

edges = sort_edges(edges)

edges = converter.convert_list_of_points(edges)

draw_edges(arm, edges, verbose=True)
