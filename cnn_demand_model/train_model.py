import logging
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

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    dataset = PointGridDataset(data, value_range, (10, 10), n_steps=1)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    print(f"Training dataset size: {len(dataset)}")


    epochs = 1
    learning_rate = 0.01
    model = DemandNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = rmse_loss

    for epoch in range(epochs):
        print(f'\n{epoch+1} pass through the full training set')

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
