from coordinates import CoordinatesConverter
from utils import sort_edges
import numpy as np
from arms import get_big_drawing_arm, calibrate, draw_edges
import skimage as ski
from pebble import concurrent

from photo2drawing import rgb2edges, edge_image2edges

arm = get_big_drawing_arm()

above_origin = np.array([250, -100, 147])
above_p1 = np.array([560, -100, 148])
above_p2 = np.array([250, 100, 147])

@concurrent.process
def make_converter(image):
    origin, p1, p2 = calibrate(arm, above_origin, above_p1, above_p2, epsilon=0.05)
    converter = CoordinatesConverter(list(reversed(image.shape[:2])), origin, p1, p2)
    return converter


@concurrent.process
def process_image(image, nb_edges=1000):

    edges = rgb2edges(image, nb_edges=nb_edges)
    edges = sort_edges(edges)
    return edges


@concurrent.process
def process_edge_image(edge_image, nb_edges=700):
    """
    Delete me
    """
    return sort_edges(edge_image2edges(edge_image))


if __name__=="__main__":

    image = ski.io.imread("data/cheat-thin.png")
    image = image > ski.filters.threshold_otsu(image)
    future_edges = process_edge_image(image)
    future_converter = make_converter(image)

    converter = future_converter.result()
    edges = future_edges.result()
    
    edges = converter.convert_list_of_points(edges)
    draw_edges(arm, edges, verbose=True)
