"""Fun with coordinates."""

import numpy as np


class CoordinatesConverter:
    def __init__(self, image_shape, origin, point1, point2):
        image_length = max(image_shape)
        image_length_idx = np.argmax(image_shape)
        image_width = min(image_shape)
        image_width_idx = np.argmin(image_shape)
        e1 = point1 - origin
        e2 = point2 - origin
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        plane_length = max(l1, l2)
        plane_length_idx = np.argmax([l1, l2])
        plane_length_vec = [e1, e2][plane_length_idx]
        plane_width = min(l1, l2)
        plane_width_idx = np.argmin([l1, l2])
        plane_width_vec = [e1, e2][plane_width_idx]

        image_ratio = image_length/image_width
        plane_ratio = plane_length/plane_width

        mat = np.zeros((2, 3))

        if image_ratio > plane_ratio:
            mat[image_length_idx] = plane_length_vec/image_length
            mat[image_width_idx] = (plane_width_vec*plane_length)/(plane_width*image_length)
        else:
            mat[image_width_idx] = plane_width_vec/image_width
            mat[image_length_idx] = (plane_length_vec*plane_width)/(plane_length*image_width)

        self.mat = mat.T
        self.origin = origin.T

    def convert(self, points):
        return np.dot(self.mat, points) + self.origin

