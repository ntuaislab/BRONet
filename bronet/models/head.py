import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class head(nn.Linear):
    """Build the head to outputs."""

    def __init__(self, num_features: int, num_classes: int, use_lln: bool) -> None:
        super(head, self).__init__(num_features, num_classes)
        self.use_lln = use_lln

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        if self.training:
            mu = x.mean(dim=0)
            x = x - mu
            self.mean += (mu - self.mean).detach() * 0.1
        else:
            x = x - self.mean
        """
        weight = self.get_weight()
        input = input.view(input.size(0), -1)
        input = F.linear(input, weight, self.bias)
        return input

    def get_weight(self):
        if self.use_lln:
            weight = F.normalize(self.weight, dim=1)
        else:
            weight = self.weight
        return weight
