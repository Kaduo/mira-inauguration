"""
Functions to turn a picture into a drawable sketch.
"""

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from edge_walker import group_edges


def rgb2edges(image, plot=False):
    """
    Convert the RGB image into a binary image with only edges.
    """
    gray_image = ski.color.rgb2gray(image)
    blurred_image = ski.restoration.denoise_bilateral(
        gray_image, sigma_color=0.05, sigma_spatial=2
    )
    edge_image = ski.feature.canny(blurred_image)
    skeletonized_image = 1 - skeletonize(edge_image)

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
        plt.show()

    return skeletonized_image


def rgb2grouped_edges(
    image,
    nb_groups=1000,
    min_edge_length=10,
    step=5,
    plot_preprocessing=False,
    plot_result=False,
):
    edge_image = rgb2edges(image, plot_preprocessing)


def grouping_edges(image, maximum_groups, rescale=True, min_edge_length=10, step=5):
    edge_image = rgb2edges(image)

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

    print(maximum)
    for i in range(maximum, 0, -1):
        if len(filtered_indexes) > maximum_groups:
            break
        for id_point_group in range(len(filtered_edge_groups)):
            if len(filtered_edge_groups[id_point_group]) == i:
                filtered_indexes.append(id_point_group)
    size_groups = len(filtered_indexes)
    print(size_groups)

    copy_filtered_edge = []
    for idx in range(len(filtered_edge_groups)):
        if idx in filtered_indexes:
            copy_filtered_edge.append(filtered_edge_groups[idx])
    filtered_edge_groups = copy_filtered_edge

    return edge_image, filtered_edge_groups


def plotting_contours(filtered_edge_groups):
    step = 2
    plt.gca().invert_yaxis()
    for point_group in filtered_edge_groups:
        preceding_point = point_group[0]
        for p in point_group[1::step]:
            plt.plot(
                (preceding_point[0], p[0]),
                (preceding_point[1], p[1]),
                c="black",
                linewidth=0.2,
            )
            preceding_point = p.copy()

    plt.gca().set_aspect("equal")
