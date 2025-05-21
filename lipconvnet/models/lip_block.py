import torch.nn as nn
from typing import Optional

from utils.misc import activation_mapping
from .layers import BRO


class LipBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        conv_layer,
        activation_name,
        stride=1,
        kernel_size=3,
        input_size: Optional[int] = 32,
        mask_level=0.125,
    ):
        super(LipBlock, self).__init__()
        if conv_layer is BRO:
            self.conv = conv_layer(in_planes, planes * stride, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, input_size=input_size, mask_level=mask_level)
        else:
            self.conv = conv_layer(in_planes, planes * stride, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.activation = activation_mapping(activation_name, planes * stride)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x


if __name__ == "__main__":
    print(LipBlock(3, 64, nn.Conv2d, 'relu'))
