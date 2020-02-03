import logging
from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from shapely.geometry import mapping
from tools import points_per_cell

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from simobility.utils import read_polygon
from demand_net import DemandNet


class DemandDataset(Dataset):
    """Dataset consists of rides "images" - to predict
    next N minutes use aggregated demand for the past N minutes.

    TODO: add static demand, e.g. average per hour per day per week

    # From the paper (https://www.dropbox.com/s/ujqova12lnklgn5/dynamic-fleet-management-TR.pdf?dl=0)
    # ..actual demand heat maps from the last two steps and constant
    # planes with sine and cosine of day of week and hour of day
    """

    def __init__(self, rides, bounding_box, image_shape: Tuple[int, int]):
        super().__init__()
        # current demand
        self.X = []
        # future demand
        self.y = []

        # to predict demand for the next N minutes
        # take N minutes of rides before
        rides_before = None
        for grp, next_rides in rides.groupby(rides.pickup_datetime):

            if rides_before is not None:
                x = points_per_cell(
                    rides_before.pickup_lon,
                    rides_before.pickup_lat,
                    bounding_box,
                    image_shape,
                )

                # im = Image.fromarray(255 - (x * (255 / x.max())).astype(np.int32))
                # im.convert('L').save('111.png')

                y = points_per_cell(
                    next_rides.pickup_lon,
                    next_rides.pickup_lat,
                    bounding_box,
                    image_shape,
                )

                self.X.append(x)
                self.y.append(y)

            rides_before = next_rides

        print(f"Dataset size: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        transform = transforms.Compose([transforms.ToTensor()])
        x = transform(x.astype(np.float32))
        y = transform(y.astype(np.float32))
        return x, y


def rmse_loss(y_pred, y):
    return torch.sqrt(torch.mean((y_pred - y) ** 2))


def train_model(data_loader, image_shape):
    learning_rate = 0.001
    max_iterations = 50

    print(f'Max training iterations {max_iterations}')

    model = DemandNet(image_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = rmse_loss

    for i, (images, labels) in enumerate(data_loader):
        outputs = model(images)

        labels = labels.view(labels.size(0), -1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == max_iterations:
            break

        if i % 100 == 0:
            print(f"Iteration {i}, training loss {loss}")

    print(f"Train loss {loss}")

    return model


def evaluate_model(model: nn.Module, data_loader):
    # TODO: implement proper validation function

    # Flatten predicted demand images and calculate RMSE

    criterion = rmse_loss
    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            # TODO: predictions must be positive integers or zeros
            predicted = model(images)
            labels = labels.view(labels.size(0), -1)

            # loss = criterion(predicted, labels)

            # print(f'Test loss {loss}')


def prepare_data_loader(rides, bounding_box, image_shape, batch_size):
    rides.pickup_datetime = rides.pickup_datetime.dt.round("10min")

    data = DemandDataset(rides, bounding_box, image_shape)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    # NOTE: demand file preprocessed using scripts from simobility
    parser.add_argument("--train-dataset", help="Feather file with trip data")
    parser.add_argument("--test-dataset", help="Feather file with trip data")
    parser.add_argument(
        "--geofence", help="Geojson file with operational area geometry"
    )
    args = parser.parse_args()

    geofence = read_polygon(args.geofence)
    # lon/lat order
    bounding_box = geofence.bounds

    train = pd.read_feather(args.train_dataset)
    test = pd.read_feather(args.test_dataset)

    batch_size = 5
    image_shape = (212, 219)

    train_loader = prepare_data_loader(train, bounding_box, image_shape, batch_size)
    test_loader = prepare_data_loader(test, bounding_box, image_shape, batch_size)

    model = train_model(train_loader, image_shape)

    # torch.save(model.state_dict(), 'demand_model.pth')

    evaluate_model(model, test_loader)
