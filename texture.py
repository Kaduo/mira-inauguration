import skimage as ski
import numpy as np
import random


def top_left_to_bottom_right(shape, nb_lines):
    texture = np.ones(shape)
    for i in range(nb_lines):
        r0 = 0
        r1 = shape[0] - 1 - i * shape[0] // nb_lines
        c0 = i * shape[1] // nb_lines
        c1 = shape[1] - 1
        rr, cc = ski.draw.line(r0, c0, r1, c1)
        texture[rr, cc] = 0

        r0 = i * shape[0] // nb_lines
        c0 = 0
        r1 = shape[0] - 1
        c1 = shape[1] - 1 - i * shape[1] // nb_lines
        rr, cc = ski.draw.line(r0, c0, r1, c1)
        texture[rr, cc] = 0

    return texture


def top_right_to_bottom_left(shape, nb_lines):
    texture = np.ones(shape)
    for i in range(nb_lines):
        r0 = i * shape[0] // nb_lines
        r1 = 0
        c0 = 0
        c1 = i * shape[1] // nb_lines
        rr, cc = ski.draw.line(r0, c0, r1, c1)
        texture[rr, cc] = 0

        r0 = shape[0] - 1
        c0 = i * shape[1] // nb_lines
        r1 = i * shape[0] // nb_lines
        c1 = shape[1] - 1
        rr, cc = ski.draw.line(r0, c0, r1, c1)
        texture[rr, cc] = 0

    return texture


def criss_cross(shape, nb_lines):
    return top_left_to_bottom_right(shape, nb_lines) * top_right_to_bottom_left(shape, nb_lines)
