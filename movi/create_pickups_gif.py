import argparse
import pandas as pd
import numpy as np
from PIL import Image
from simobility.utils import read_polygon

from tools import points_per_cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create demand gif")
    parser.add_argument("--dataset", help="Feather file with trip data")
    parser.add_argument(
        "--geofence", help="Geojson file with operational area geometry"
    )
    args = parser.parse_args()

    geofence = read_polygon(args.geofence)
    # lon/lat order
    bounding_box = geofence.bounds

    rides = pd.read_feather(args.dataset)

    rides.pickup_datetime = rides.pickup_datetime.dt.round("10min")

    image_shape = (212, 219)
    num_frames = 100

    frames = []
    for grp, items in rides.groupby(rides.pickup_datetime):
        counts = points_per_cell(
            items.pickup_lon, items.pickup_lat, bounding_box, image_shape,
        )

        im = Image.fromarray(255 - (counts * (255 / counts.max())).astype(np.int32))
        # im = im.convert("L")
        frames.append(im)

        if len(frames) == num_frames:
            break

    frames[0].save(
        "aggregated_pickups.gif",
        save_all=True,
        append_images=frames[1:],
        # optimize=False,
        duration=100,
        loop=0,
    )

