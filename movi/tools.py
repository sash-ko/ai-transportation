from typing import Tuple
import numpy as np


def points_per_cell(
    x_vals: np.array,
    y_vals: np.array,
    values_range: Tuple[Tuple[float, float]],
    output_shape: Tuple[int, int],
):
    """ Creates are grid of 'output_shape' size and counts
    the number of points per each cell

    Params
    ------

    x_vals: np.array - an array of floats, e.g. pickups lon
    y_vals: np.array - an array of floats
    values_range: max and min values for x and y axis
    output_shape: shape of the greed - number of rows and columns

    NOTE: x_vals/y_vals is a subset of values inside values_range. For example,
    values_range can be a bounding box of a whole city and x_vals/y_vals are
    points inside a some neighborhood
    """
    (min_x, min_y, max_x, max_y) = values_range

    x_axis = np.linspace(min_x, max_x, output_shape[0])
    y_axis = np.linspace(min_y, max_y, output_shape[1])

    x_pixel_idx = np.digitize(x_vals, x_axis)
    y_pixel_idx = np.digitize(y_vals, y_axis)


    image, _, _ = np.histogram2d(
        x_pixel_idx,
        y_pixel_idx,
        bins=output_shape,
        range=[[0, output_shape[0]], [0, output_shape[1]]],
    )
    return image
