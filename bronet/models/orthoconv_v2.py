import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .cholesky_grad import CholeskyOrthConv


class OrthoConvV2(nn.Module):
    """
    A backbone version that keeps the padded idx in the output feature map of each layer.
    These region contain information from the central region after circular convolution.
    The output feature map will increase by 2 * (kernel_size // 2) in each layer.
    """

    def __init__(
        self,
        width: int,
        depth: int,
        input_size: int,
        activation: nn.Module,
        mode: str,
        weight_rank_ratio: float = 0.5,
        quasi_residual: bool = True,
        id_random_init: bool = False,
        zero_padding: bool = True,
    ) -> None:
        super(OrthoConvV2, self).__init__()
        self.mode = mode
        self.kernel_size = 3
        self.act = activation

        self.depth = depth
        self.width = width
        self.scale = depth**-0.5
        self.quasi_residual = quasi_residual

        if mode == "bro":
            weight_rank = self.weight_rank_mapping(width, weight_rank_ratio)
        else:
            weight_rank = width
        weights = torch.randn(depth, width, weight_rank, self.kernel_size, self.kernel_size)
        self.weight_rank = weight_rank

        weights = weights / (width * self.kernel_size**2)

        if mode == "bro":
            identity = torch.zeros(width, weight_rank, self.kernel_size, self.kernel_size)
            identity[:weight_rank, :weight_rank, 1, 1] = torch.eye(weight_rank)
        else:
            identity = torch.zeros(width, width, self.kernel_size, self.kernel_size)
            identity[:width, :width, 1, 1] = torch.eye(width)
        self.register_buffer("identity", identity)

        self.id_random_init = id_random_init
        if self.id_random_init:
            assert self.quasi_residual is False
            weights = self.scale * weights + self.identity

        self.weights = nn.Parameter(weights)

        self.zero_padding = zero_padding
        self.padding_input_sizes = []
        padding_input_size = input_size
        if self.zero_padding:
            for _ in range(depth):
                padding_input_size += (self.kernel_size // 2) * 2
                self.padding_input_sizes.append(padding_input_size)
        else:
            self.padding_input_sizes = [input_size] * depth
        gamma = torch.ones(depth, width, 1, 1, 1)
        self.gamma = nn.Parameter(gamma)

        self.bias = nn.Parameter(torch.zeros(depth, width))
        running_mean = torch.zeros(depth, width)
        self.register_buffer("running_mean", running_mean)

        I_low = torch.eye(weight_rank, requires_grad=False).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.register_buffer("I_low", I_low)
        I_full = torch.eye(width, requires_grad=False).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.register_buffer("I_full", I_full)

    def forward(self, x):
        if self.training:
            weights = self.get_wfft_ortho()
            all_means = []
            for i, (weight, bias) in enumerate(zip(weights, self.bias)):
                z = self.frequency_convolution(x, weight, i)
                mean = z.mean((0, 2, 3))
                all_means.append(mean.detach())
                z = z + (bias - mean).view(-1, 1, 1)
                x = self.act(z)
            all_means = torch.stack(all_means)
            self.running_mean += (all_means - self.running_mean) * 0.1
            return x

        with torch.no_grad():
            weights = self.eval_weights
            biases = self.bias - self.running_mean
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                z = self.frequency_convolution(x, weight, i)
                z = z + bias.view(-1, 1, 1)
                x = self.act(z)
        return x

    def frequency_convolution(self, x, weight, layer_idx):
        if self.zero_padding:
            x = F.pad(x, (self.kernel_size // 2,) * 4)
        padded_n = self.kernel_size // 2
        batches, _, n, _ = x.shape
        if n != self.padding_input_sizes[layer_idx]:
            raise ValueError(f"Input size {n} does not match the expected size {self.padding_input_sizes[layer_idx]}")
        xfft = torch.fft.rfft2(x, norm="ortho").permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), self.width, batches)
        zfft = weight @ xfft
        zfft = zfft.reshape(n, n // 2 + 1, self.width, batches).permute(3, 2, 0, 1)
        z = torch.fft.irfft2(zfft, norm="ortho").real

        # z = z[:, :, padded_n:-padded_n, padded_n:-padded_n]
        return z

    @staticmethod
    def fft_shift_matrix(n, shift_amount):
        # NOTE: for moving of the kernel center to [0,0]
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * shift_amount * shift / n)

    def process_vfft(self, V_i, padding_input_size, ortho_func):
        """Process an individual V_i tensor."""
        # NOTE: Shifting slightly works better in v2 (i.e. when not removing the padded idx) as it preserves the feature locality better.
        shift_amount = (self.kernel_size - 1) // 2
        shift_matrix = self.fft_shift_matrix(padding_input_size, -shift_amount)[:, : (padding_input_size // 2 + 1)].to(
            V_i.device
        )
        VFFT_i = shift_matrix * torch.fft.rfft2(V_i, (padding_input_size, padding_input_size), norm="ortho").conj()
        VFFT_i = VFFT_i.permute(2, 3, 0, 1)  # Rearrange dimensions for mode processing

        W_FFT_i_ortho = ortho_func(VFFT_i)  # Example: Call BRO function
        return W_FFT_i_ortho.reshape(
            padding_input_size * (padding_input_size // 2 + 1),
            self.width,
            self.width,
        )

    def get_wfft_ortho(self):
        if self.quasi_residual:
            V = self.identity + self.weights * self.gamma * self.scale
        else:
            V = self.gamma * self.weights

        WFFTs = []
        for V_i, padding_input_size in zip(V, self.padding_input_sizes):
            assert self.mode == "bro"  # Other ortho parameterizations are not supported yet
            WFFT_i = self.process_vfft(V_i, padding_input_size, self.BRO)
            WFFTs.append(WFFT_i)

        return WFFTs

    def lipschitz(self):
        return 1.0

    def extra_repr(self) -> str:
        return f"{self.mode}: depth={self.depth}, width={self.width}, weight_rank={self.weight_rank}, quasi_residual={self.quasi_residual}, id_random_init={self.id_random_init}, zero_padding={self.zero_padding}"

    def BRO(self, VFFT):
        assert VFFT.shape[-2] > VFFT.shape[-1]
        V_T_V_FFT = VFFT.transpose(-2, -1).conj() @ VFFT
        eps = V_T_V_FFT.diagonal(dim1=-2, dim2=-1).mean(dim=-1).mul(1e-7)
        V_T_V_FFT = V_T_V_FFT + eps.view(*eps.shape, 1, 1) * self.I_low
        W_FFT_ortho = self.I_full - 2 * VFFT @ torch.linalg.solve(V_T_V_FFT, VFFT.transpose(-2, -1).conj())
        return W_FFT_ortho

    def train(self, mode=True):
        self.training = mode
        if mode is False:
            with torch.no_grad():
                weights = self.get_wfft_ortho()
                self.eval_weights = weights
        else:
            if hasattr(self, "eval_weights"):
                del self.eval_weights
        return self

    @staticmethod
    def weight_rank_mapping(width, weight_rank_ratio):
        if weight_rank_ratio == 1.0:
            return width - 1
        elif weight_rank_ratio == 0.0:
            return 1
        else:
            weight_rank = int(width * weight_rank_ratio)
        assert weight_rank > 0 and weight_rank <= width
        return weight_rank
