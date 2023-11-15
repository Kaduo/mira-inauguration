"""Functions to group edge pixels by walking along them."""

from enum import Enum
import itertools
import numpy as np


class Direction(Enum):
    Left = 0
    TopLeft = 1
    Top = 2
    TopRight = 3
    Right = 4
    BottomRight = 5
    Bottom = 6
    BottomLeft = 7


NB_DIRS = 8


def next_clockwise(dir):
    return Direction((dir.value + 1) % NB_DIRS)


def directions(starting_dir=Direction.Left):
    dir = starting_dir
    for _ in range(NB_DIRS):
        yield dir
        dir = next_clockwise(dir)


def neighbor(x, y, dir):
    match dir:
        case Direction.Left | Direction.TopLeft | Direction.BottomLeft:
            dx = -1
        case Direction.Right | Direction.TopRight | Direction.BottomRight:
            dx = +1
        case _:
            dx = 0

    match dir:
        case Direction.Top | Direction.TopLeft | Direction.TopRight:
            dy = -1
        case Direction.Bottom | Direction.BottomLeft | Direction.BottomRight:
            dy = +1
        case _:
            dy = 0

    return [x + dx, y + dy]


def opposite_dir(dir):
    return Direction((dir.value + 4) % NB_DIRS)


def neighbors(x, y, start_dir=Direction.Left):
    for dir in directions(start_dir):
        yield neighbor(x, y, dir)


def move_along_edge(start_x, start_y, is_edge, starting_dir=Direction.Left):
    current_dir = starting_dir
    go_on = True
    x = start_x
    y = start_y
    while go_on:
        go_on = False
        for dir, neighbor in zip(
            directions(current_dir),
            itertools.islice(neighbors(x, y, current_dir), NB_DIRS - 1),
        ):
            if is_edge(neighbor[0], neighbor[1]):
                x = neighbor[0]
                y = neighbor[1]
                current_dir = next_clockwise(opposite_dir(dir))
                yield dir, [x, y]
                go_on = True
                break


def whole_edge(x, y, is_edge):
    res = [[x, y]]
    first = True
    first_dir = None
    for dir, point in move_along_edge(x, y, is_edge):
        if first:
            first_dir = dir
            first = False
        res.append(point)
        if point in res[:-1]:
            break

    count = 0
    if first_dir is not None:
        for dir, point in move_along_edge(x, y, is_edge, starting_dir=next_clockwise(first_dir)):
            if point in res:
                break
            else:
                count += 1
                res.insert(0, point)

    return res


def group_edges(edge_image, min_edge_length=2, step=1):
    edges = []
    width = edge_image.shape[1]
    height = edge_image.shape[0]

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def is_edge(x, y):
        return in_bounds(x, y) and edge_image[y, x] == 0 and visited[y, x] == 1

    visited = np.full(edge_image.shape, 1)
    diff = visited != edge_image
    where_diff = np.where(diff)

    while len(where_diff[0]) > 0:
        print(len(where_diff[0]))
        p = where_diff[0][0], where_diff[1][0]
        y, x = p[0], p[1]
        new_edge = whole_edge(x, y, is_edge)
        if len(new_edge) >= min_edge_length:
            edges.append(np.array(new_edge[::step] + [new_edge[-1]]))
            print(edges[-1].shape)

        for p in new_edge:
            visited[p[1], p[0]] = 0

        diff = visited != edge_image
        where_diff = np.where(diff)

    return edges


if __name__ == "__main__":
    x, y = 0, 0
    for n in neighbors(x, y):
        continue
