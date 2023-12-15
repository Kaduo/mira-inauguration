"""
Functions to turn a picture into a drawable sketch.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from edge_walker import group_edges
import matplotlib.colors as mcolors
from itertools import cycle
from fdog import difference_of_gaussians


def rgb2edge_image(image, plot=False, bilateral=True, method="canny"):
    """
    Convert an RGB image into a binary image of its edges.
    """
    try:
        gray_image = ski.color.rgb2gray(image)
    except IndexError:
        gray_image = ski.color.rgb2gray(ski.color.rgba2rgb(image))
    if bilateral:
        blurred_image = ski.restoration.denoise_bilateral(
            gray_image, sigma_color=0.05, sigma_spatial=1.5
        )
    else:
        blurred_image = gray_image

    if method == "canny":
        edge_image = ski.feature.canny(blurred_image)
        skeletonized_image = 1 - skeletonize(edge_image)

    elif method == "dog":
        print("TODO ! (:")

    if plot:
        _, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3)
        ax1.imshow(image)
        ax1.set_title("1. Original image.")
        ax2.imshow(gray_image)
        ax2.set_title("2. Gray image.")
        ax3.imshow(blurred_image)
        ax3.set_title("3. Bilateral filter.")
        ax4.imshow(edge_image)
        ax4.set_title("4. Canny filter.")
        ax5.imshow(skeletonized_image)
        ax5.set_title("5. Skeletonize filter.")

    return skeletonized_image


def rgb2edges(
    image,
    nb_edges=1000,
    min_edge_length=10,
    step=5,
    plot_preprocessing=False,
    plot_result=False,
    return_edge_image=False,
    method="canny",
):
    edge_image = rgb2edge_image(image, plot_preprocessing, method=method)

    res = edge_image2edges(
        edge_image,
        min_edge_length=min_edge_length,
        step=step,
        nb_edges=nb_edges,
        plot_result=plot_result,
        plot_preprocessing=plot_preprocessing,
    )

    if return_edge_image:
        return res, edge_image
    else:
        return res


def edge_image2edges(
    edge_image,
    min_edge_length=10,
    step=5,
    nb_edges=1000,
    plot_result=False,
    plot_preprocessing=False,
):
    edges = group_edges(edge_image, min_edge_length=min_edge_length, step=step)

    # Sort edges by length
    sorted_edges = sorted(edges, key=len, reverse=True)

    # Select the nb_edges longest edges
    res = sorted_edges[:nb_edges]

    smallest_length = len(res[-1])

    # Also select every edge with the same length as the shortest edge already selected
    for edge in sorted_edges[nb_edges:]:
        if len(edge) < smallest_length:
            break
        else:
            res.append(edge)

    if plot_result:
        plot_edges(res)

    return res


def grouping_edges(image, maximum_groups, rescale=True, min_edge_length=10, step=5):
    edge_image = rgb2edge_image(image)

    max_length = max(edge_image.shape[0], edge_image.shape[1])
    edge_groups = group_edges(edge_image)
    edge_group_lens = []
    for g in edge_groups:
        edge_group_lens.append(len(g))

    if rescale:
        max_length = max(image.shape[0], image.shape[1])
    else:
        max_length = 1

    filtered_edge_groups = []

    for point_group in edge_groups:
        if len(point_group) >= min_edge_length:
            filtered_edge_groups.append(point_group[::step].copy() / max_length)

    maximum = np.max(edge_group_lens)
    filtered_indexes = []

    for i in range(maximum, 0, -1):
        if len(filtered_indexes) > maximum_groups:
            break
        for id_point_group in range(len(filtered_edge_groups)):
            if len(filtered_edge_groups[id_point_group]) == i:
                filtered_indexes.append(id_point_group)
    size_groups = len(filtered_indexes)

    copy_filtered_edge = []
    for idx in range(len(filtered_edge_groups)):
        if idx in filtered_indexes:
            copy_filtered_edge.append(filtered_edge_groups[idx])
    filtered_edge_groups = copy_filtered_edge

    return edge_image, filtered_edge_groups


def plot_edges(edges, use_different_colors=True):
    if use_different_colors:
        colors = mcolors.XKCD_COLORS.values()
    else:
        colors = ["black"]

    plt.figure()
    plt.title("Edges")

    plt.gca().invert_yaxis()
    for edge, color in zip(edges, cycle(colors)):
        preceding_point = edge[0]
        for p in edge[1:]:
            plt.plot(
                (preceding_point[0], p[0]),
                (preceding_point[1], p[1]),
                c=color,
                linewidth=0.2,
            )
            preceding_point = p.copy()

    plt.gca().set_aspect("equal")


def rgb2dog_edge_image(image, low_sigma, high_sigma=None, p=0, thresh_technique="mean"):
    if high_sigma is None:
        high_sigma = low_sigma * 1.7

    dog_image = difference_of_gaussians(image, low_sigma, high_sigma, p=p)
    match thresh_technique:
        case "otsu":
            thresh_image = dog_image > ski.filters.threshold_otsu(dog_image)
        case _:
            thresh_image = dog_image > ski.filters.threshold_mean(dog_image)

    dilated_image = ski.morphology.dilation(thresh_image)

    abs_diff = 1 - thresh_image ^ dilated_image

    return abs_diff
