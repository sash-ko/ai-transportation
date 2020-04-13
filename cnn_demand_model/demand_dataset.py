import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PointGridDataset(Dataset):
    """ Dataset for spatio-temporal prediction of demand - predict future demand (number of points
    per cell) using current demand and demand several steps before. For example, predict demand
    20 minutes from now knowing current demand, demand 20 and 40 minutes ago.
    
    The dataset aggregates points by time and maps to a grid.
    """

    def __init__(
        self,
        points: pd.DataFrame,
        value_range: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_size: Tuple[int, int],
        n_steps: int = 1,
    ):
        """
        Params
        ------

        points : pd.DataFrame
            Contains x, y coordinates and time. Expected schema: [x, y, time]
        
        value_range : tuple
            Range of x and y values: [[xmin, xmax], [ymin, ymax]]

        grid_size : tuple
            Number of x and y cells

        n_steps: int
            Number of time steps to be used as features. Determines the feature shape:
            'n_steps x grid_size[0] x grid_size[1]'
        """

        super().__init__()
        # current demand
        self.X = []
        # future demand
        self.y = []

        self.n_steps = n_steps

        current_X = []
        for grp, point_grp in points.groupby(["time"]):
            x = point_grp.x.values
            y = point_grp.y.values
            hist, _, _ = np.histogram2d(x, y, bins=grid_size, range=value_range)

            if len(current_X) < n_steps:
                current_X.append(hist)
            else:
                self.X.append(np.array(current_X))
                self.y.append(hist)

                current_X.append(hist)
                current_X.pop(0)

        logging.info(f"Dataset size: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        return x, y
