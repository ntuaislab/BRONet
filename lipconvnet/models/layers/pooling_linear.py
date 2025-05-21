import numpy as np
import torch.nn as nn

class PoolingLinear(nn.Module):

    def __init__(self, ncin, ncout, agg="mean"):
        super(PoolingLinear, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.agg = agg

    def forward(self, x):
        if self.agg == "trunc":
            return x[:, :self.ncout]
        k = 1. * self.ncin / self.ncout
        out = x[:, :self.ncout * int(k)]
        out = out.view(x.shape[0], self.ncout, -1)
        if self.agg == "mean":
            out = np.sqrt(k) * out.mean(axis=2)
        elif self.agg == "max":
            out, _ = out.max(axis=2)
        return out