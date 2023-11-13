from coordinates import CoordinatesConverter
from utils import sort_edges
import numpy as np
from arms import get_big_drawing_arm, calibrate, draw_edges
import skimage as ski
from pebble import concurrent

from photo2drawing import rgb2edges

arm = get_big_drawing_arm()

above_origin = np.array([200, -100, 147])
above_p1 = np.array([560, -100, 148])
above_p2 = np.array([200, 135, 147])

@concurrent.process
def make_converter(image):
    origin, p1, p2 = calibrate(arm, above_origin, above_p1, above_p2, epsilon=1)
    converter = CoordinatesConverter(list(reversed(image.shape[:2])), origin, p1, p2)
    return converter


@concurrent.process
def process_image(image, nb_edges=1000):

    edges = rgb2edges(image, nb_edges=nb_edges)
    edges = sort_edges(edges)
    return edges


if __name__=="__main__":
    image = ski.io.imread("data/st jerome.jpg")
    future_edges = process_image(image)
    future_converter = make_converter(image)

    converter = future_converter.result()
    edges = future_edges.result()
    
    edges = converter.convert_list_of_points(edges)
    draw_edges(arm, edges, verbose=True)
