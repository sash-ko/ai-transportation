from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from shapely.geometry import mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from simobility.utils import read_polygon


class Net(nn.Module):

    """
    The output of the network is a 212×219 image in which each 
    pixel stands for the predicted number of ride requests in 
    a given region in the next 30 minutes

    The network inputs six feature planes whose size is 212×219: 
    actual demand heat maps from the last two steps and constant 
    planes with sine and cosine of day of week and hour of day
    """

    def __init__(self):
        super(Net, self).__init__()

        # The first hidden layer convolves 16 filters of 5×5
        self.conv1 = nn.Conv2d(6, 16, 5)

        # The second layers convolves 32 filters of 3×3
        self.conv2 = nn.Conv2d(16, 32, 3)

        # The final layer convolves 1 filter of kernel size 1×1
        self.conv3 = nn.Conv2d(32, 1, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x


def rides_to_image(
    pickup_lat: np.array,
    pickup_lon: np.array,
    bounds: Tuple[Tuple[float, float]],
    image_shape: Tuple[int, int],
) -> np.array:
    """ Create NxM (image_shape) image in which each pixel stands for the predicted
    number of ride requests in a given region in the next 30 minutes

    Params
    ------

    pickup_lat :
    pickup_lon :
    bounds : service area bounding box ((min_lon, min_lat), (max_lon, max_lat))
    image_shape : 

    Returns

    image : NxM (image_shape) array in which each element is a total 
    number of pickups in a an area defined by the cell

    """

    x_axis = np.linspace(bounds[0], bounds[2], image_shape[0])
    y_axis = np.linspace(bounds[1], bounds[3], image_shape[1])

    x_pixel_idx = np.digitize(pickup_lon, x_axis)
    y_pixel_idx = np.digitize(pickup_lat, y_axis)

    image, _, _ = np.histogram2d(
        x_pixel_idx,
        y_pixel_idx,
        bins=image_shape,
        range=[[0, image_shape[0]], [0, image_shape[1]]],
    )
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--demand-file", help="Feather file with trip data")
    parser.add_argument(
        "--geofence", help="Geojson file with operational area geometry"
    )
    args = parser.parse_args()

    rides = pd.read_feather(args.demand_file)
    geofence = read_polygon(args.geofence)
    bounds = geofence.bounds

    image_shape = (212, 219)

    image = rides_to_image(rides.pickup_lon, rides.pickup_lat, bounds, image_shape)
