import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from typing import Optional


class BRO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        input_size: Optional[int] = 32,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        mask_level: float = 1,
        id_init: bool = True,
        ortho_init: bool = False,
    ) -> None:
        if stride == 2:
            self.stride = 2
            in_channels = in_channels * self.stride**2
        else:
            self.stride = 1

        self.raw_in_channels = in_channels
        self.raw_out_channels = out_channels
        self.in_channels = self.mask_mapping(in_channels, mask_level) if in_channels == out_channels else in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.input_size = input_size
        self.padded_input_size = self.input_size // self.stride + (kernel_size // 2) * 2

        self.max_channels = max(in_channels, out_channels)

        super().__init__()

        self.weight = nn.Parameter(torch.empty((self.out_channels, self.in_channels, kernel_size, kernel_size)))

        if self.weight.shape[1] > self.weight.shape[0]:
            self.weight.data = self.weight.data.permute(1, 0, 2, 3).contiguous()

        self.min_channels = min(self.weight.shape[0], self.weight.shape[1])

        self.I = torch.eye(self.max_channels, dtype=torch.complex64, requires_grad=False).to(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        self.I_m = (
            torch.eye(self.min_channels, dtype=torch.complex64, requires_grad=False)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.id_init = id_init
        self.ortho_init = ortho_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.id_init:
            N, M, K, K2 = self.weight.shape
            assert K == K2
            self.weight.data.zero_()
            self.weight.data[np.arange(min(N, M)), np.arange(min(N, M)), K // 2, K // 2] = 1.0
        elif self.ortho_init:
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def lipschitz(self):
        return 1.0

    def fft_shift_matrix(self, n, s):
        """
        If the padding is symmetric, a shfit matrix is needed
        """
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    def cal_wfft_ortho(self, n, cout, cin):
        assert self.weight.shape[1] < self.weight.shape[0]
        vfft_tensor = torch.fft.rfft2(self.weight, (n, n), norm="ortho")

        vfft_tensor = vfft_tensor.permute(2, 3, 0, 1)
        wfft_tensor = vfft_tensor.transpose(2, 3).conj() @ vfft_tensor

        perturb_ratio = 1e-5
        eps = wfft_tensor.diagonal(dim1=-2, dim2=-1).mean(dim=-1).mul(perturb_ratio)
        wfft_tensor = wfft_tensor + eps.view(*eps.shape, 1, 1) * self.I_m
        wfft_ortho = self.I - 2 * vfft_tensor @ torch.linalg.solve(wfft_tensor, vfft_tensor.transpose(2, 3).conj())

        wfft_ortho = wfft_ortho[:, :, :cout, :cin]
        wfft_ortho = wfft_ortho.reshape(n * (n // 2 + 1), cout, cin)

        # n*n, cout, cin
        return wfft_ortho

    def get_weight(self):
        W = self.cal_wfft_ortho(self.padded_input_size, self.raw_out_channels, self.raw_in_channels)
        return W

    def train(self, mode=True):
        self.training = mode
        if mode is False:
            W = self.get_weight().detach()
            self.eval_weight = W
        return self

    @staticmethod
    def mask_mapping(in_channels, mask_level):
        assert mask_level >= 0 and mask_level <= 1
        if mask_level == 0:
            return 1
        elif mask_level == 0.25:
            return in_channels // 4
        elif mask_level == 0.5:
            return in_channels // 2
        elif mask_level == 0.75:
            return in_channels // 4 * 3
        elif mask_level == 1:
            return in_channels - 1
        else:
            return int(in_channels * mask_level)

    def forward(self, x):
        if self.stride > 1:
            x = einops.rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )

        padded_n = 0
        assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1]
        if self.kernel_size[0] > 1:  # zero-pad
            x = F.pad(x, (self.kernel_size[0] // 2,) * 4)
            padded_n += self.kernel_size[0] // 2

        cout, cin = self.raw_out_channels, self.raw_in_channels
        batches, _, n, _ = x.shape  # (bs, cin, n, n)
        # (note that n would be affected by padding and strd)
        if n != self.padded_input_size:
            raise ValueError(f"Padded Input size {n} does not match the expected size {self.padded_input_size}")

        xfft = torch.fft.rfft2(x, norm="ortho").permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)

        if self.training:
            W = self.get_weight()
        else:
            W = self.eval_weight

        zfft = W @ xfft

        zfft = zfft.reshape(n, (n // 2 + 1), cout, batches).permute(3, 2, 0, 1)
        z = torch.fft.irfft2(zfft, norm="ortho").real if zfft.shape[3] > 1 else zfft.real

        if padded_n > 0:
            z = z[:, :, padded_n:-padded_n, padded_n:-padded_n]
        if self.bias is not None:
            z += self.bias[:, None, None]

        return z

    def extra_repr(self):
        return f"V_col_rank={self.in_channels}, id_init={self.id_init}, ortho_init={self.ortho_init}"


class BROLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module, id_init: bool = True, ortho_init: bool = False, mask_level: float = 0.5) -> None:
        super(BROLinear, self).__init__()
        max_features = max(in_features, out_features)
        if id_init:
            weights = torch.zeros(1, max_features, self.mask_mapping(max_features, mask_level))
            weights[:, : self.mask_mapping(max_features, mask_level), :] = torch.eye(self.mask_mapping(max_features, mask_level))
        elif ortho_init:
            weights = torch.randn(1, max_features, self.mask_mapping(max_features, mask_level))
            weights = torch.nn.init.orthogonal_(weights)
        else:
            weights = torch.randn(1, max_features, self.mask_mapping(max_features, mask_level))
            weights = weights / max_features
        self.id_init = id_init
        self.ortho_init = ortho_init
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.zeros(1, max_features))

        self.register_buffer("identity", torch.eye(max_features))

        self.register_buffer("identity_half", torch.eye(self.mask_mapping(max_features, mask_level)))
        self.register_buffer("eval_weights", torch.zeros(1, max_features, max_features))

        self.act = activation
        self.width = max_features

    @staticmethod
    def mask_mapping(in_channels, mask_level):
        assert mask_level >= 0 and mask_level <= 1
        if mask_level == 0:
            return 1
        elif mask_level == 0.25:
            return in_channels // 4
        elif mask_level == 0.5:
            return in_channels // 2
        elif mask_level == 0.75:
            return in_channels // 4 * 3
        elif mask_level == 1:
            return in_channels - 1
        else:
            return int(in_channels * mask_level)

    def get_weights(self):
        V = self.weights
        Sigma = V.transpose(1, 2) @ V
        eps = Sigma.diagonal(dim1=1, dim2=2).mean(-1).mul(1e-5)
        Sigma = Sigma + eps.view(*eps.shape, 1, 1) * self.identity_half.unsqueeze(0)
        W = self.identity - 2 * V @ torch.linalg.solve(Sigma, V.transpose(1, 2))
        return W

    def forward(self, x):
        if self.training:
            weights = self.get_weights()
        else:
            weights = self.eval_weights
        for weight, bias in zip(weights, self.bias):
            x = F.linear(x, weight, bias)
            x = self.act(x)
        return x

    def train(self, mode=True):
        self.training = mode
        if mode is False:
            weights = self.get_weights().detach()
            self.eval_weights += weights - self.eval_weights
        return self

    def lipschitz(self):
        return 1.0

    def extra_repr(self) -> str:
        return f"CPHH_rfft: depth={1}, width = {self.width}, mode=half, id_init = {self.id_init}, ortho_init = {self.ortho_init}"


if __name__ == "__main__":
    for mask_level, cin, cout in [(1, 32, 16), (1, 32, 32), (1, 128, 128), (1, 256, 256)]:
        all_diff = []
        print(f"{cin=}, {cout=}, {mask_level=}")
        x = torch.randn(16, cin, 32, 32).cuda()
        y = torch.randn(16, cin, 32, 32).cuda()
        bro = BRO(
            in_channels=cin,
            out_channels=cout,
            kernel_size=3,
            input_size=32,
            mask_level=mask_level,
            id_init=False,
        ).cuda()
        print(bro)
        bro.eval()
        for i in range(x.shape[0]):
            z = bro(x).real
            zz = bro(y).real
            inputData = x[i, :, :, :] - y[i, :, :, :]
            outputData = z[i, :, :, :] - zz[i, :, :, :]
            all_diff.append(outputData.norm().item() / inputData.norm().item())
        print("mean: ", np.mean(all_diff))
        print("max: ", np.max(all_diff))
        print("std: ", np.std(all_diff))
#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永無BUG
#
