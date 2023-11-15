from coordinates import CoordinatesConverter
from utils import sort_edges
import numpy as np
from arms import get_big_drawing_arm, calibrate, draw_edges
import skimage as ski
from pebble import concurrent
import pickle

from photo2drawing import rgb2edges, edge_image2edges

arm = get_big_drawing_arm()

above_origin = np.array([226, -129, 155])
above_p1 = np.array([600, -129, 155])
above_p2 = np.array([226, 126, 160])

@concurrent.process
def make_converter(image):
    origin, p1, p2 = calibrate(arm, [above_origin, above_p1, above_p2], absolute_epsilon = [1, 1, 0.5])
    converter = CoordinatesConverter(list(reversed(image.shape[:2])), origin, p1, p2)
    return converter


@concurrent.process
def process_image(image, nb_edges=1000):

    edges = rgb2edges(image, nb_edges=nb_edges)
    edges = sort_edges(edges)
    return edges


if __name__=="__main__":

    image = ski.io.imread("data/famille.jpeg")
    # future_edges = process_image(image)
    future_converter = make_converter(image)

    converter = future_converter.result()

    with open("family_edges.pkl", "rb") as f:
        edges = pickle.load(f)
        f.close()
    
    edges = converter.convert_list_of_points(edges)
    draw_edges(arm, edges, verbose=True)
