import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

"""

Paper "MOVI: A Model-Free Approach to Dynamic Fleet Management": 
https://www.dropbox.com/s/ujqova12lnklgn5/dynamic-fleet-management-TR.pdf?dl=0

Propose spatial-temporal demand prediction approach with CNNs only...

    > The output of the network is a 212×219 image in which each 
    > pixel stands for the predicted number of ride requests in 
    > a given region in the next 30 minutes

    > The network inputs six feature planes whose size is 212×219: 
    > actual demand heat maps from the last two steps and constant 
    > planes with sine and cosine of day of week and hour of day

There are some other research papers that propose the same, e.g.
- "Data-DrivenMulti-step Demand Prediction for Ride-hailingServices Using Convolutional Neural Network"
    https://arxiv.org/pdf/1911.03441.pdf

- "Forecasting Taxi Demands with Fully ConvolutionalNetworks and Temporal Guided Embedding"
    https://openreview.net/pdf?id=BygF00DuiX

This paper has more citations:

- "Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction" 
    https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16069/15978

TODO: look at this topics deeper
"""

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
