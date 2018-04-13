
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.autograd import Variable

class WeightedL1(nn.Module):
    # Weighted L1 Loss function for pytorch
    def __init__(self):
        super(WeightedL1, self).__init__()

    def forward(self, x, target, w):
        return (w * torch.abs(x-target)).mean()

class WeightedL2(nn.Module):
    # Weighted L2 Loss function for pytorch
    def __init__(self):
        super(WeightedL2, self).__init__()

    def forward(self, x, target, w):
        return torch.sum(w * (x-target) ** 2)


