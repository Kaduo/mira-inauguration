"""Fun with coordinates."""

import numpy as np


def find_short_and_long_side(shape):
    long_idx = np.argmax(shape)
    short_idx = np.argmin(shape)

    return short_idx, long_idx


class CoordinatesConverter:
    """
    Arguments:
    image_shape -- the dimensions of the original image: (height, width)
    origin, point1, point2 -- numpy arrays of shape (3, 1)
    """

    def __init__(self, image_shape, origin, point1, point2):
        image_short_idx, image_long_idx = find_short_and_long_side(image_shape)

        image_long_length = max(image_shape)
        image_short_length = min(image_shape)

        e1 = point1 - origin
        e2 = point2 - origin
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        plane_short_idx, plane_long_idx = find_short_and_long_side([l1, l2])
        plane_long_length = max(l1, l2)
        plane_long_vec = [e1, e2][plane_long_idx]

        plane_short_length = min(l1, l2)
        plane_short_vec = [e1, e2][plane_short_idx]

        image_ratio = image_long_length / image_short_length
        plane_ratio = plane_long_length / plane_short_length

        mat = np.zeros((2, 3))

        if image_ratio > plane_ratio:
            mat[image_long_idx] = plane_long_vec / image_long_length
            mat[image_short_idx] = (plane_short_vec * plane_long_length) / (
                plane_short_length * image_long_length
            )
        else:
            mat[image_short_idx] = plane_short_vec / image_short_length
            mat[image_long_idx] = (plane_long_vec * plane_short_length) / (
                plane_long_length * image_short_length
            )

        self.mat = mat.T
        self.origin = origin.T

    def convert(self, points):
        """
        Arguments:
        points -- numpy array of shape (2, number_of_points)

        Returns a numpy array of shape (3, number_of_points)
        """
        return np.dot(self.mat, points) + self.origin

    def convert_list_of_points(self, list_of_points):
        """
        Argument:
        list_of_points -- list of numpy arrays of shape (number_of_points, 2)
        """
        res = []

        for points in list_of_points:
            res.append(self.convert(points.T))

        return res
