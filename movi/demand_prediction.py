from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from shapely.geometry import mapping
from h3 import h3

from simobility.utils import read_polygon


def rides_to_image(
    pickup_lat: np.array,
    pickup_lon: np.array,
    bounds: Tuple[Tuple[float]],
    image_shape: Tuple[int],
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
