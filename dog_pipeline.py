from fdog import difference_of_gaussians
import skimage as ski
from photo2drawing import edge_image2edges


def rgb2dog_edge_image(image, low_sigma, high_sigma=None, p=0, thresh_technique="mean"):
    if high_sigma is None:
        high_sigma = low_sigma * 1.7

    dog_image = difference_of_gaussians(image, low_sigma, high_sigma, p=p)
    match thresh_technique:
        case "otsu":
            thresh_image = dog_image > ski.filters.threshold_otsu(dog_image)
        case "mean":
            thresh_image = dog_image > ski.filters.threshold_mean(dog_image)
        case n:
            thresh_image = dog_image > n

    dilated_image = ski.morphology.dilation(thresh_image)

    abs_diff = 1 - thresh_image ^ dilated_image

    return abs_diff


def dog_edge_image2edges(dog_edge_image, min_edge_length=10, step=5, nb_edges=700):
    return edge_image2edges(dog_edge_image, min_edge_length, step, nb_edges)


def rgb2dog_edges(
    image,
    low_sigma,
    high_sigma=None,
    p=0,
    thresh_technique="mean",
    min_edge_length=10,
    step=5,
    nb_edges=700,
):
    dog_edge_image = rgb2dog_edge_image(image, low_sigma, high_sigma, p, thresh_technique)
    return dog_edge_image2edges(dog_edge_image, min_edge_length, step, nb_edges)
