import tomllib
from arms import get_arm, calibrate, draw_edges, wait_for_input
import skimage as ski
from photo2drawing import rgb2edge_image, rgb2edges
from photomaton import photomaton_loop
from coordinates import CoordinatesConverter
from utils import sort_edges
import signal
import cv2


def load_config():
    with open("config.toml", "rb") as f:
        return tomllib.load(f)


def signal_handler(sig, frame):
    wait_for_input("continue")


def calibrate_from_config(arm, config):
    origin, p1, p2 = calibrate(
        arm,
        [
            config["calibration"]["above_origin"],
            config["calibration"]["above_p1"],
            config["calibration"]["above_p2"],
        ],
        absolute_epsilon=config["calibration"]["epsilon"],
    )

    return origin, p1, p2


def get_coordinates_converter(arm, config, image_shape):
    # Calibrate robot
    origin, p1, p2 = calibrate_from_config(arm, config)

    # Create converter
    converter = CoordinatesConverter(image_shape, origin, p1, p2)

    return converter


def edges_from_config(image, config):
    edges = rgb2edges(
        image,
        nb_edges=config["edge_processing"]["nb_edges"],
        min_edge_length=config["edge_processing"]["min_edge_length"],
        step=config["edge_processing"]["edge_step"],
        method=config["image_processing"]["edge_finding_method"],
    )

    if config["edge_processing"]["sort_edge"]:
        edges = sort_edges(edges)

    return edges


def draw_image(arm, image, config, converter=None):
    if converter is None:
        converter = get_coordinates_converter(arm, config, image.shape)
    edges = edges_from_config(image, config)
    converted_edges = converter.convert_list_of_points(edges)
    draw_edges(arm, converted_edges)


def photomaton_meta_loop(arm, config):
    cap = cv2.VideoCapture(config["photomaton"]["camera_index"])
    converter = None
    while True:
        if config["photomaton"]["recalibrate_every_time"]:
            converter = None
        image = photomaton_loop(cap)
        draw_image(arm, image, config, converter)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    config = load_config()

    # Connect to robot
    arm = get_arm(config["robot_ip"])

    if config["enable_photomaton"]:
        photomaton_meta_loop(arm, config)

    else:
        image = ski.io.imread(config["image_path"])
        draw_image(arm, image, config)
