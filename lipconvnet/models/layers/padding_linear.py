import torch
import torch.nn as nn
import numpy as np

class PaddingChannels(nn.Module):

    def __init__(self, ncout, ncin=3, mode="zero"):
        super(PaddingChannels, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

    def forward(self, x):
        if self.mode == "clone":
            return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
        elif self.mode == "zero":
            bs, _, size1, size2 = x.shape
            out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
            out[:, :self.ncin] = x
            return out


