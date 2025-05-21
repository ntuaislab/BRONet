import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedLinear(nn.Linear):
    def forward(self, input):
        input = input.view(input.shape[0], -1)
        self.input = input
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self.lln_weight = self.weight / weight_norm
        return F.linear(
            input,
            self.lln_weight if self.training else self.lln_weight.detach(),
            self.bias,
        )
