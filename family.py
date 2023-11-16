from coordinates import CoordinatesConverter
from utils import sort_edges
import numpy as np
from arms import get_big_drawing_arm, calibrate, draw_edges, wait_for_input, draw_edge
import skimage as ski
from pebble import concurrent
import pickle
from time import time
import signal
import sys


def signal_handler(sig, frame):
    wait_for_input("continue")


from photo2drawing import rgb2edges, edge_image2edges

arm = get_big_drawing_arm()

above_origin = np.array([220, -123, 142])
above_p1 = np.array([527, -123, 146])
above_p2 = np.array([220, 131, 142])


@concurrent.process
def make_converter(image):
    origin, p1, p2 = calibrate(
        arm, [above_origin, above_p1, above_p2], absolute_epsilon=[1, 1, 1]
    )
    converter = CoordinatesConverter(list(reversed(image.shape[:2])), origin, p1, p2)
    return converter


@concurrent.process
def process_image(image, nb_edges=1000):
    edges = rgb2edges(image, nb_edges=nb_edges)
    edges = sort_edges(edges)
    return edges


if __name__=="__main__":

    with open("data/jerome 2 edges.pkl", "rb") as f:
        edges = pickle.load(f)
        print("HEEREE")
        print(len(edges))
        f.close()
    
    # 13h16

    signal.signal(signal.SIGINT, signal_handler)
    image = ski.io.imread("data/st jerome.jpg")
    # future_edges = process_image(image)
    future_converter = make_converter(image)

    converter = future_converter.result()

    # Draw border
    border_edge = np.array(
        [
            converter.origin,
            converter.p1,
            converter.p1 + (converter.p2 - converter.origin),
            converter.p2,
            converter.origin
        ]
    ).T[0]
    print(border_edge.shape)
    draw_edge(arm, border_edge)
    wait_for_input("sfdkggienc")


    start = time()
    edges = converter.convert_list_of_points(edges)
    draw_edges(arm, edges, verbose=True)

    end = time()

    print(end - start)
