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
        print(f"Epoch \n{epoch}")

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


if __name__ == "__main__":
    # Example:
    # python cnn_demand_model/train_model.py --dataset data/train_sample.feather
    #   --value-range="((-74.0238037109375, -73.91867828369139), (40.6966552734375, 40.81862258911133))"

    parser = argparse.ArgumentParser(description="Model training parameters")
    parser.add_argument("--dataset", help="feather file with points")
    parser.add_argument(
        "--value-range", help="Bounding box - (min_lon, max_lon, min_lat, max_lat)"
    )
    args = parser.parse_args()

    #### Params
    grid_size = (50, 50)
    batch_size = 5
    agg_by = "10min"

    # combination of "use_threads=False" and specified columns works faster than
    # all other methods
    data = pd.read_feather(
        args.dataset,
        columns=["pickup_lon", "pickup_lat", "pickup_datetime"],
        use_threads=False,
    )
    print(f"Dataset shape {data.shape}")

    data["time"] = data.pickup_datetime.dt.round(agg_by)
    data["x"] = data.pickup_lon
    data["y"] = data.pickup_lat

    # parse value range from command line
    min_lon, max_lon, min_lat, max_lat = re.findall("[-]?\d+.\d+", args.value_range)
    value_range = ((float(min_lon), float(max_lon)), (float(min_lat), float(max_lat)))
    print(f"Bounding box: {value_range}")

    dataset = PointGridDataset(data, value_range, grid_size, n_steps=1)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training dataset size: {len(dataset)}")

    model = DemandNet(grid_size)

    criterion = rmse_loss
    train(model, data_loader, criterion)
