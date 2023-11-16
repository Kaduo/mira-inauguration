from coordinates import CoordinatesConverter
from utils import sort_edges
import numpy as np
from arms import get_big_drawing_arm, calibrate, draw_edges, wait_for_input, draw_edge
import skimage as ski
from pebble import concurrent
import pickle
import signal
import sys


def signal_handler(sig, frame):
    wait_for_input("continue")


from photo2drawing import rgb2edges, edge_image2edges

arm = get_big_drawing_arm()

above_origin = np.array([270, -107, 160])
above_p1 = np.array([630, -107, 162])
above_p2 = np.array([270, 158, 157])


@concurrent.process
def make_converter(image):
    origin, p1, p2 = calibrate(
        arm, [above_origin, above_p1, above_p2], absolute_epsilon=[1, 1, 0.5]
    )
    converter = CoordinatesConverter(list(reversed(image.shape[:2])), origin, p1, p2)
    return converter


@concurrent.process
def process_image(image, nb_edges=1000):
    edges = rgb2edges(image, nb_edges=nb_edges)
    edges = sort_edges(edges)
    return edges


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    image = ski.io.imread("data/st jerome.jpg")
    # future_edges = process_image(image)
    future_converter = make_converter(image)

    converter = future_converter.result()

    with open("data/jerome 2 edges.pkl", "rb") as f:
        edges = pickle.load(f)
        f.close()

    # Draw border
    border_edge = np.array(
        [
            converter.origin,
            converter.p1,
            converter.p1 + (converter.p2 - converter.origin),
            converter.p2,
        ]
    ).T
    draw_edge(arm, border_edge)
    wait_for_input("sfdkggienc")

    edges = converter.convert_list_of_points(edges)
    draw_edges(arm, edges, verbose=True)
