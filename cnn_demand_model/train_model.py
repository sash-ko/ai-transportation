import logging
from math import ceil
import argparse
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from demand_dataset import PointGridDataset


class DemandNet(nn.Module):

    """
    Spatial-temporal demand prediction
    """

    def __init__(self, input_shape):
        super(DemandNet, self).__init__()

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
