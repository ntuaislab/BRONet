import torch.nn as nn
from .layers import Conv2d, Sequential, build_activation
from typing import Tuple


def build_stem(
    input_size: int,
    width: int,
    act_name: str,
    kernel_size: int = 5,
) -> Tuple[nn.Module, int]:
    """Build the first layer to inputs."""
    conv_f = Conv2d
    if input_size == 64:  # Tiny ImageNet
        conv = conv_f(3, width, kernel_size=kernel_size, stride=2, input_size=64)
        output_size = 32
    elif input_size == 32:  # CIFAR10/100
        conv = conv_f(3, width, kernel_size=kernel_size, stride=2, input_size=32)
        output_size = 16
    elif input_size == 224:
        patch_size = 14
        # patch_size = round((width / 3) ** 0.5)
        conv = conv_f(
            3,
            width,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            input_size=224,
        )
        output_size = 224 // patch_size
    else:
        raise ValueError("Unsupported `input_size`!")

    activation = build_activation(act_name, dim=1, channels=width)
    stem_layer = Sequential(conv, activation)
    return stem_layer, output_size
