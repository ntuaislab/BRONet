import torch.nn as nn
from typing import Optional

from models.layers.soc import SOC

from .layers import (
    BRO,
    NormalizedLinear,
    CayleyLinear,
)


from utils.misc import (
    conv_mapping,
    activation_mapping,
)
from .lip_block import LipBlock


class LipConvNet(nn.Module):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, num_classes=10, input_size=64, lln=False, kernel_size=3, num_dense=0, mask_level=0.125):
        super(LipConvNet, self).__init__()
        self.lln = lln
        self.in_planes = 3
        self.num_dense = num_dense

        conv_layer = conv_mapping[conv_name]
        assert isinstance(block_size, int)

        if conv_layer is BRO or conv_layer is SOC:
            self.layer1 = self._make_layer(init_channels, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size, input_size=input_size, mask_level=mask_level)
            input_size //= 2
            self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size, input_size=input_size, mask_level=mask_level)
            input_size //= 2
            self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size, input_size=input_size, mask_level=mask_level)
            input_size //= 2
            self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size, input_size=input_size, mask_level=mask_level)
            input_size //= 2
            self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=1, input_size=input_size, mask_level=mask_level)
            input_size //= 2
        else:
            self.layer1 = self._make_layer(init_channels, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size)
            self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size)
            self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size)
            self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=kernel_size)
            self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer, activation, stride=2, kernel_size=1)
            input_size //= 32  # 1

        flat_size = input_size
        flat_features = flat_size * flat_size * self.in_planes  # 1024
        self.avg_pool = nn.AvgPool2d(flat_size)
        self.dense_layers = []
        for _ in range(num_dense):
            self.dense_layers.append(BRO(flat_features, flat_features, activation_mapping(activation, flat_features)))
            # flat_features = flat_features // 2
        self.dense_layers = nn.Sequential(*self.dense_layers)
        if num_dense > 0 and not self.lln:
            raise NotImplementedError("Currently dense layer only supports lln.")

        if self.lln:
            self.last_layer = NormalizedLinear(flat_features, num_classes)
        elif conv_name == "cayley":
            self.last_layer = CayleyLinear(flat_features, num_classes)
        else:
            self.last_layer = conv_layer(flat_features, num_classes, kernel_size=1, stride=1)

    def _make_layer(
        self,
        planes,
        num_blocks,
        conv_layer,
        activation,
        stride,
        kernel_size,
        input_size: Optional[int] = False,
        mask_level=0.125,
    ):
        strides = [1] * (num_blocks - 1) + [stride]
        kernel_sizes = [3] * (num_blocks - 1) + [kernel_size]
        layers = []
        for _, (stride, kernel_size) in enumerate(zip(strides, kernel_sizes)):
            if conv_layer is BRO:
                layers.append(LipBlock(self.in_planes, planes, conv_layer, activation, stride, kernel_size, input_size=input_size, mask_level=mask_level))
            else:
                layers.append(LipBlock(self.in_planes, planes, conv_layer, activation, stride, kernel_size=kernel_size))
            self.in_planes = planes * stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.num_dense > 0:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dense_layers(x)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x

    def set_and_get_lipschitz_constant(self):
        L = 1.0
        if self.training:
            return L
        for _, module in self.named_modules():
            if hasattr(module, "norm_bound"):
                sigma = module.norm_bound(module.input_size)
                L = L * sigma
        self.lip_const = L
        return L
