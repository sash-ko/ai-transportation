import logging

import argparse
import pandas as pd
from torch.utils.data import DataLoader
import torch

from demand_dataset import PointGridDataset
from demand_net import DemandNet


def rmse_loss(y_pred, y):
    return torch.sqrt(torch.mean((y_pred - y) ** 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training parameters")
    parser.add_argument("--dataset", help="feather file with points")
    args = parser.parse_args()

    data = pd.read_feather(args.dataset, use_threads=True)
    data["time"] = data.pickup_datetime.dt.round("10min")
    data["x"] = data.pickup_lon
    data["y"] = data.pickup_lat

    value_range = ((data.x.min(), data.x.max()), (data.y.min(), data.y.max()))

    #### Params
    grid_size = (10, 10)
    batch_size = 5
    epochs = 2
    learning_rate = 0.01

    dataset = PointGridDataset(data, value_range, grid_size, n_steps=1)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training dataset size: {len(dataset)}")

    model = DemandNet(grid_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = rmse_loss

    for epoch in range(epochs):
        print(f"\n{epoch+1} pass through the full training set")

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
