import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import einops
from torch.autograd.grad_mode import no_grad


def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T


def l2_normalize(tensor):
    tensor = F.normalize(tensor.flatten(start_dim=0), dim=0).view_as(tensor)
    return tensor


class SOC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        input_size=32,
        padding=None,
        bias=True,
        train_terms=6,
        eval_terms=15,
        init_iters=50,
        update_iters=1,
        multiplier=1.0,
    ):
        super(SOC, self).__init__()
        assert (stride == 1) or (stride == 2)
        self.input_size = input_size
        self.padding = padding

        self.forward_func = self.conv_exp

        self.init_iters = init_iters
        self.update_iters = update_iters

        self.out_channels = out_channels
        self.in_channels = in_channels * stride * stride

        self.max_channels = max(self.out_channels, self.in_channels)

        diff_channels = max(0, self.out_channels - self.in_channels)
        self.pad_channels = (0, 0, 0, 0, 0, diff_channels, 0, 0)

        self.stride = stride
        self.kernel_size = kernel_size

        self.train_terms = train_terms
        self.eval_terms = eval_terms

        self.pad_side = self.kernel_size // 2

        if kernel_size == 1:
            multiplier = 1.0
        self.multiplier = multiplier

        self.weight = nn.Parameter(
            torch.randn(
                self.max_channels, self.max_channels, self.kernel_size, self.kernel_size
            ),
            requires_grad=True,
        )
        self.enable_bias = bias
        if self.enable_bias:
            self.bias = nn.Parameter(torch.rand(self.out_channels), requires_grad=True)
        self.reset_parameters()

        self._initialize_singular_vectors()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.max_channels)
        nn.init.normal_(self.weight, std=stdv)

        if self.enable_bias:
            stdv = 1.0 / np.sqrt(self.max_channels)
            nn.init.uniform_(self.bias, -stdv, stdv)

    def _initialize_singular_vectors(self):
        num_ch, k_size = self.max_channels, self.kernel_size

        u1 = self.weight.new_empty((1, num_ch, 1, k_size)).normal_(0, 1)
        self.register_buffer("_u1", l2_normalize(u1))

        v1 = self.weight.new_empty((num_ch, 1, k_size, 1)).normal_(0, 1)
        self.register_buffer("_v1", l2_normalize(v1))

        u2 = self.weight.new_empty((1, num_ch, k_size, 1)).normal_(0, 1)
        self.register_buffer("_u2", l2_normalize(u2))

        v2 = self.weight.new_empty((num_ch, 1, 1, k_size)).normal_(0, 1)
        self.register_buffer("_v2", l2_normalize(v2))

        u3 = self.weight.new_empty((1, num_ch, k_size, k_size)).normal_(0, 1)
        self.register_buffer("_u3", l2_normalize(u3))

        v3 = self.weight.new_empty((num_ch, 1, 1, 1)).normal_(0, 1)
        self.register_buffer("_v3", l2_normalize(v3))

        u4 = self.weight.new_empty((num_ch, 1, k_size, k_size)).normal_(0, 1)
        self.register_buffer("_u4", l2_normalize(u4))

        v4 = self.weight.new_empty((1, num_ch, 1, 1)).normal_(0, 1)
        self.register_buffer("_v4", l2_normalize(v4))

    @no_grad()
    def _power_method(self, skew_filter, num_iters=50):
        for i in range(num_iters):
            self._v1 = l2_normalize((skew_filter * self._u1).sum((1, 3), keepdim=True))
            self._u1 = l2_normalize((skew_filter * self._v1).sum((0, 2), keepdim=True))

            self._v2 = l2_normalize((skew_filter * self._u2).sum((1, 2), keepdim=True))
            self._u2 = l2_normalize((skew_filter * self._v2).sum((0, 3), keepdim=True))

            self._v3 = l2_normalize(
                (skew_filter * self._u3).sum((1, 2, 3), keepdim=True)
            )
            self._u3 = l2_normalize((skew_filter * self._v3).sum(0, keepdim=True))

            self._v4 = l2_normalize(
                (skew_filter * self._u4).sum((0, 2, 3), keepdim=True)
            )
            self._u4 = l2_normalize((skew_filter * self._v4).sum(1, keepdim=True))

    def update_sigma(self):
        weight_T = transpose_filter(self.weight)
        skew_filter = 0.5 * (self.weight - weight_T)

        self._power_method(skew_filter, num_iters=self.init_iters)

    def _train_update_sigma(self):
        if self.training:
            weight_T = transpose_filter(self.weight)
            skew_filter = 0.5 * (self.weight - weight_T)

            self._power_method(skew_filter, num_iters=self.update_iters)

    def compute_skew_filter(self):
        weight_T = transpose_filter(self.weight)
        skew_filter = 0.5 * (self.weight - weight_T)

        s1 = torch.sum(skew_filter * self._u1 * self._v1)
        s2 = torch.sum(skew_filter * self._u2 * self._v2)
        s3 = torch.sum(skew_filter * self._u3 * self._v3)
        s4 = torch.sum(skew_filter * self._u4 * self._v4)

        sigma = torch.minimum(torch.minimum(s1, s2), torch.minimum(s3, s4))
        skew_filter = (self.multiplier * skew_filter) / sigma
        return skew_filter

    def conv_exp(self, x, skew_filter, n):
        z = x
        for i in range(n, 0, -1):
            z = x + (F.conv2d(z, skew_filter, padding=self.pad_side) / i)
        return z

    def forward(self, x):
        self._train_update_sigma()
        if self.training:
            num_terms = self.train_terms
        else:
            num_terms = self.eval_terms

        skew_filter = self.compute_skew_filter()

        if self.stride > 1:
            x = einops.rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )

        if self.out_channels > self.in_channels:
            x = F.pad(x, self.pad_channels)

        z = self.forward_func(x, skew_filter, num_terms - 1)

        if self.out_channels < self.in_channels:
            z = z[:, : self.out_channels, :, :]

        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z

    @no_grad()
    def initial_input(self, input_size):
        if self.kernel_size == 1:
            input_size = self.stride
        stride = self.stride
        in_channels = self.in_channels // (stride * stride)

        u = self.weight.new_empty((1, in_channels, input_size, input_size)).normal_(
            0, 1
        )
        u = l2_normalize(u)

        if stride > 1:
            u = einops.rearrange(
                u, "b c (w k1) (h k2) -> b (c k1 k2) w h", k1=stride, k2=stride
            )
        if self.out_channels > self.in_channels:
            u = F.pad(u, self.pad_channels)
        return u

    def lipschitz(self):
        return self.norm_bound(self.input_size)

    @no_grad()
    def norm_bound(self, input_size):
        u = self.initial_input(input_size)
        skew_filter = self.compute_skew_filter()

        for i in range(self.init_iters):
            v = F.conv2d(u, skew_filter, padding=self.pad_side)
            v = l2_normalize(v)

            u = F.conv2d(v, -skew_filter, padding=self.pad_side)
            u = l2_normalize(u)

        v = F.conv2d(u, skew_filter, padding=self.pad_side)
        sigma_skew = torch.norm(v, p=2).item()

        error_bound = np.power(sigma_skew, self.eval_terms) / np.math.factorial(
            self.eval_terms
        )
        sigma = 1.0 + error_bound
        return sigma

    @no_grad()
    def norm(self, input_size):
        u = self.initial_input(input_size)
        skew_filter = self.compute_skew_filter()

        for i in range(self.init_iters):
            v = self.forward_func(u, skew_filter, self.eval_terms - 1)
            v = l2_normalize(v)

            u = self.forward_func(v, -skew_filter, self.eval_terms - 1)
            u = l2_normalize(u)

        v = self.forward_func(u, skew_filter, self.eval_terms - 1)
        sigma = torch.norm(v, p=2).item()
        return sigma

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
        )


if __name__ == "__main__":
    model = SOC(3, 3, 3)
    print(model)
    print(model.lipschitz())
    model = SOC(512, 512, 3)
    print(model)
    print(model.lipschitz())
