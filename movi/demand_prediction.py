import logging
from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from math import ceil
from shapely.geometry import mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from simobility.utils import read_polygon


class Net(nn.Module):

    """
    The output of the network is a 212×219 image in which each 
    pixel stands for the predicted number of ride requests in 
    a given region in the next 30 minutes

    The network inputs six feature planes whose size is 212×219: 
    actual demand heat maps from the last two steps and constant 
    planes with sine and cosine of day of week and hour of day

    Source: https://www.dropbox.com/s/ujqova12lnklgn5/dynamic-fleet-management-TR.pdf?dl=0
    """

    def __init__(self, input_shape):
        super(Net, self).__init__()

        # The first hidden layer convolves 16 filters of 5×5
        self.conv1 = nn.Conv2d(1, 16, 5)
        output_shape = (
            calc_out_size(input_shape[0], 5),
            calc_out_size(input_shape[1], 5),
        )

        # The second layers convolves 32 filters of 3×3
        self.conv2 = nn.Conv2d(16, 32, 3)
        output_shape = (
            calc_out_size(output_shape[0], 3),
            calc_out_size(output_shape[1], 3),
        )
        # The final layer convolves 1 filter of kernel size 1×1
        self.conv3 = nn.Conv2d(32, 1, 1)

        output_shape = (
            calc_out_size(output_shape[0], 1),
            calc_out_size(output_shape[1], 1),
        )

        self.fc = nn.Linear(output_shape[0] * output_shape[1], 100)

        # back to the original image size
        self.fc2 = nn.Linear(100, input_shape[0] * input_shape[1])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return x


def calc_out_size(in_size: int, kernel_size: int, padding: int = 0, stride: int = 1):
    """Calculate output size of any dimention"""
    return ceil((in_size - kernel_size + 2 * padding) / stride + 1)


def rides_to_image(
    pickup_lat: np.array,
    pickup_lon: np.array,
    bounding_box: Tuple[Tuple[float, float]],
    image_shape: Tuple[int, int],
) -> np.array:
    """ Create NxM (image_shape) image in which each pixel stands for the predicted
    number of ride requests in a given region in the next 30 minutes

    Params
    ------

    pickup_lat :
    pickup_lon :
    bounding_box : service area bounding box ((min_lon, min_lat), (max_lon, max_lat))
    image_shape : 

    Returns

    image : NxM (image_shape) array in which each element is a total 
    number of pickups in a an area defined by the cell

    """

    x_axis = np.linspace(bounding_box[0], bounding_box[2], image_shape[0])
    y_axis = np.linspace(bounding_box[1], bounding_box[3], image_shape[1])

    x_pixel_idx = np.digitize(pickup_lon, x_axis)
    y_pixel_idx = np.digitize(pickup_lat, y_axis)

    image, _, _ = np.histogram2d(
        x_pixel_idx,
        y_pixel_idx,
        bins=image_shape,
        range=[[0, image_shape[0]], [0, image_shape[1]]],
    )
    return image


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
                x = rides_to_image(
                    rides_before.pickup_lon,
                    rides_before.pickup_lat,
                    bounding_box,
                    image_shape,
                )

                y = rides_to_image(
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

    model = Net(image_shape)

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

    print(f"Training loss {loss}")

    return model


def evaluate_model(model: nn.Module, data_loader):
    # TODO: implement proper validation function

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            # TODO: predictions must be positive integers or zeros
            predicted = model(images)

            labels = labels.view(labels.size(0), -1)


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
