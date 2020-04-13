import logging
import re
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch

from demand_dataset import PointGridDataset
from demand_net import DemandNet


def rmse_loss(y_pred, y):
    return torch.sqrt(torch.mean((y_pred - y) ** 2))


def train(model, data_loader, criterion):
    epochs = 2
    learning_rate = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch}")

        train_loss = []
        for i, data in enumerate(data_loader):
            inputs, labels = data

            outputs = model(inputs)

            labels = labels.view(labels.size(0), -1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if i % 100 == 0:
                print(
                    f"[{epoch}, {i:4d}] training loss: {np.mean(train_loss[-100:]):.3f}"
                )
        print(f"Training loss: {np.mean(train_loss):.3f}")


def test(model, data_loader, criterion):
    with torch.no_grad():
        test_loss = []

        for i, data in enumerate(data_loader):
            images, labels = data
            outputs = model(images)

            labels = labels.view(labels.size(0), -1)
            loss = criterion(outputs, labels)

            test_loss.append(loss)

            if i % 100 == 0:
                print(f"[{i:4d}] testing loss: {np.mean(test_loss[-100:]):.3f}")

        print(f"Testing loss: {np.mean(test_loss):.3f}")


def prepare_data(file_name):
    data = pd.read_feather(
        file_name,
        columns=["pickup_lon", "pickup_lat", "pickup_datetime"],
        use_threads=False,
    )
    print(f"\nDataset shape: {data.shape} ({file_name})")

    data["time"] = data.pickup_datetime.dt.round(agg_by)
    data["x"] = data.pickup_lon
    data["y"] = data.pickup_lat

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training parameters")
    parser.add_argument("--train-dataset", help="feather file with points")
    parser.add_argument("--test-dataset", help="feather file with points")
    parser.add_argument(
        "--value-range", help="Bounding box - (min_lon, max_lon, min_lat, max_lat)"
    )
    args = parser.parse_args()

    # params
    grid_size = (50, 50)
    batch_size = 5
    agg_by = "10min"

    # parse value range from command line
    min_lon, max_lon, min_lat, max_lat = re.findall("[-]?\d+.\d+", args.value_range)
    value_range = ((float(min_lon), float(max_lon)), (float(min_lat), float(max_lat)))
    print(f"Bounding box: {value_range}")

    # train data
    train_data = prepare_data(args.train_dataset)
    train_data = PointGridDataset(train_data, value_range, grid_size, n_steps=1)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = DemandNet(grid_size)

    criterion = rmse_loss

    print(f'\nTraining model')
    train(model, train_data_loader, criterion)

    # test data
    test_data = prepare_data(args.test_dataset)
    test_data = PointGridDataset(test_data, value_range, grid_size, n_steps=1)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print(f'\nTesting model')
    test(model, test_data_loader, criterion)

