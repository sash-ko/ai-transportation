import logging
from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from shapely.geometry import mapping
from tools import points_per_cell
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

from simobility.utils import read_polygon
from demand_net import DemandNetMOVI, DemandNet
from demand_prediction import (DemandDataset,
                               DemandDatasetMOVI,
                               DemandDatasetMOVIVariablePlanes,
                               rmse_loss
)

"""
Based on research paper "MOVI: A Model-Free Approach to Dynamic Fleet Management"

Predict demand using CNN where input is total number pickups per cell of a city grid
("demand image").

TODO: change input a 3D matrix where each image is a demand aggregation for N minuts bucket
in order to be able to catch some temporal information. Similar to the approach for following
research paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

def evaluate_model(model: nn.Module, data_loader):
    criterion = rmse_loss
    model.eval()
    test_loss = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            predicted = model(images)

            labels = labels.view(labels.size(0), -1)
            loss = criterion(predicted, labels)

            test_loss.append(loss.item())

    print(f'\nTest RMSE={np.mean(test_loss):.4}, RMSE std={np.std(test_loss):.4}')


def evaluate_model_MOVI(model: nn.Module, data_loader):
    criterion = rmse_loss
    model.eval()
    test_loss = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images,cd,sd,ch,sh,labels = data
            predicted = model(images,cd,sd,ch,sh)

            labels = labels.view(labels.size(0), -1)
            loss = criterion(predicted, labels)

            test_loss.append(loss.item())

    print(f'\nTest RMSE={np.mean(test_loss):.4}, RMSE std={np.std(test_loss):.4}')
    
def prepare_data_loader(rides, bounding_box, image_shape, batch_size):
    rides.rename(columns={'tpep_pickup_datetime':'pickup_datetime','pickup_latitude':
                          'pickup_lat', 'pickup_longitude': 'pickup_lon'}, inplace=True)
    rides.pickup_datetime = pd.to_datetime(rides['pickup_datetime'])

    rides.pickup_datetime = rides.pickup_datetime.dt.round("10min")

    #TODO: preprocess data!

    #data = DemandDatasetMOVIVariablePlanes(rides, bounding_box, image_shape)
    #data = DemandDatasetMOVI(rides, bounding_box, image_shape)
    data = DemandDataset(rides, bounding_box, image_shape)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    # NOTE: demand file preprocessed using scripts from simobility
    parser.add_argument("--model", help="Model file")
    parser.add_argument("--test-dataset", help="Feather file with trip data")
    parser.add_argument(
        "--geofence", help="Geojson file with operational area geometry"
    )
    args = parser.parse_args()

    geofence = read_polygon(args.geofence)
    # lon/lat order
    bounding_box = geofence.bounds

    test = pd.read_feather(args.test_dataset, use_threads=True)
    batch_size = 10
    image_shape = (212, 219)

    test_loader = prepare_data_loader(test, bounding_box, image_shape, batch_size)

    #model = DemandNetMOVI(image_shape)
    model = DemandNet(image_shape)
    model.load_state_dict(torch.load(args.model))

    #torch.save(model.state_dict(), 'demand_model_212_219_variable.pth')

    evaluate_model(model, test_loader)
    #evaluate_model_MOVI(model, test_loader)
