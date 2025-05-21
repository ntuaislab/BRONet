import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .cholesky_grad import CholeskyOrthConv
from .soc import SOC
from .sll import SDPBasedLipschitzConvLayer


class LiResConv(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        input_size: int,
        activation: nn.Module,
        centering: bool = True,
        num_lc_iter: int = 10,
    ) -> None:
        super(LiResConv, self).__init__()
        weights = torch.randn(depth, width, width, 3, 3)
        weights = weights / (width * 9)
        self.weights = nn.Parameter(weights)

        gamma = torch.ones(depth, width, 1, 1, 1)
        self.gamma = nn.Parameter(gamma)

        self.bias = nn.Parameter(torch.zeros(depth, width))
        if centering:
            running_mean = torch.zeros(depth, width)
            self.register_buffer("running_mean", running_mean)
        else:
            self.running_mean = 0

        identity = torch.zeros(width, width, 3, 3)
        identity[:, :, 1, 1] = torch.eye(width)
        identity = torch.stack([identity for _ in range(depth)])
        self.register_buffer("identity", identity)

        init_x = torch.ones(1, depth * width, *_pair(input_size))
        self.register_buffer("init_x", init_x)

        self.act = activation

        self.depth = depth
        self.width = width
        self.scale = depth**-0.5
        self.num_lc_iter = num_lc_iter
        self.centering = centering

    def get_weight(self):
        W = self.weights * self.gamma
        return self.identity + W * self.scale

    def forward(self, x):
        weights = self.get_weight()
        if not (self.centering and self.training):
            biases = self.bias - self.running_mean
            for weight, bias in zip(weights, biases):
                x = F.conv2d(x, weight, bias, padding=1)
                x = self.act(x)
            return x

        weights = weights - self.identity
        all_means = []
        for weight, bias in zip(weights, self.bias):
            out = F.conv2d(x, weight, padding=1)
            mean = out.mean((0, 2, 3))
            all_means.append(mean.detach())
            out = out + (bias - mean).view(-1, 1, 1)
            x = self.act(x + out)

        all_means = torch.stack(all_means)
        self.running_mean += (all_means - self.running_mean) * 0.1
        return x

    def lipschitz(self):
        W = self.get_weight().reshape(-1, self.width, 3, 3)
        x = self.init_x.data
        for _ in range(self.num_lc_iter):
            x = F.conv2d(x, W, padding=1, groups=self.depth)
            x = F.conv_transpose2d(x, W, padding=1, groups=self.depth)
            x = x.reshape(self.depth, -1)
            x = F.normalize(x, dim=1)
            x = x.reshape(self.init_x.shape)

        x = x.detach()

        self.init_x += (x - self.init_x).detach()
        x = F.conv2d(x, W, padding=1, groups=self.depth)
        norm = x.reshape(self.depth, -1).norm(dim=1)
        return norm.prod()

    def extra_repr(self) -> str:
        return f"depth={self.depth}, " f"width={self.width}, " f"centering={self.centering}"


class LiResConvOrtho(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        input_size: int,
        activation: nn.Module,
        mode: str,
        centering: bool = True,
        weight_rank_ratio: float = 0.25,
        quasi_residual: bool = True,
        id_random_init: bool = False,
        zero_padding: bool = True,
        shift: bool = False,
    ) -> None:
        super(LiResConvOrtho, self).__init__()
        self.mode = mode
        self.kernel_size = 3
        self.act = activation

        self.depth = depth
        self.width = width
        self.scale = depth**-0.5
        self.centering = centering
        self.quasi_residual = quasi_residual
        self.zero_padding = zero_padding

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

        if self.zero_padding:
            self.padding_input_size = input_size + (self.kernel_size // 2) * 2
        else:
            self.padding_input_size = input_size
        gamma = torch.ones(depth, width, 1, 1, 1)
        self.gamma = nn.Parameter(gamma)

        self.bias = nn.Parameter(torch.zeros(depth, width))
        if centering:
            running_mean = torch.zeros(depth, width)
            self.register_buffer("running_mean", running_mean)
        else:
            self.running_mean = 0

        I_low = torch.eye(weight_rank, requires_grad=False).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.register_buffer("I_low", I_low)
        # self.I_low = I_low
        I_full = torch.eye(width, requires_grad=False).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.register_buffer("I_full", I_full)
        self.shift = shift
        if shift:
            shift_amount = (self.kernel_size - 1) // 2
            shift_matrix = self.fft_shift_matrix(self.padding_input_size, -shift_amount)[
                :, : (self.padding_input_size // 2 + 1)
            ]
            self.shift_matrix = shift_matrix
            # self.register_buffer("shift_matrix", shift_matrix)

    @staticmethod
    def fft_shift_matrix(n, shift_amount):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * shift_amount * shift / n)

    def forward(self, x):
        if self.training:
            weights = self.get_wfft_ortho()
        else:
            weights = self.eval_weights
        if self.centering and self.training:
            all_means = []
            for weight, bias in zip(weights, self.bias):
                z = self.frequency_convolution(x, weight)
                mean = z.mean((0, 2, 3))
                all_means.append(mean.detach())
                z = z + (bias - mean).view(-1, 1, 1)
                x = self.act(z)
            all_means = torch.stack(all_means)
            self.running_mean += (all_means - self.running_mean) * 0.1
            return x

        biases = self.bias - self.running_mean
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            z = self.frequency_convolution(x, weight)
            z = z + bias.view(-1, 1, 1)
            x = self.act(z)
        return x

    def frequency_convolution(self, x, weight):
        padded_n = 0
        if self.zero_padding:
            x = F.pad(x, (self.kernel_size // 2,) * 4)
            padded_n = self.kernel_size // 2
        batches, _, n, _ = x.shape
        if n != self.padding_input_size:
            raise ValueError(f"Input size {n} does not match the expected size {self.padding_input_size}")
        xfft = torch.fft.rfft2(x, norm="ortho").permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), self.width, batches)
        zfft = weight @ xfft
        zfft = zfft.reshape(n, n // 2 + 1, self.width, batches).permute(3, 2, 0, 1)
        z = torch.fft.irfft2(zfft, norm="ortho").real

        if self.zero_padding:
            z = z[:, :, padded_n:-padded_n, padded_n:-padded_n]
        return z

    def get_wfft_ortho(self):

        # quasi-residual trick from LiResNet++ cholesky dense layer, LOT needs it to be stabilized.
        if self.quasi_residual:
            V = self.identity + self.weights * self.gamma * self.scale
        else:
            V = self.gamma * self.weights

        # print(V.shape)
        if self.shift:
            if self.shift_matrix.device != V.device:
                self.shift_matrix = self.shift_matrix.to(V.device)
            VFFT = (
                self.shift_matrix
                * torch.fft.rfft2(V, (self.padding_input_size, self.padding_input_size), norm="ortho").conj()
            )
        else:
            VFFT = torch.fft.rfft2(V, (self.padding_input_size, self.padding_input_size), norm="ortho")
        VFFT = VFFT.permute(0, 3, 4, 1, 2)

        if self.mode == "bro":
            W_FFT_ortho = self.BRO(VFFT)
        elif self.mode == "lot":
            W_FFT_ortho = self.LOT_ortho(VFFT)
        elif self.mode == "cholesky":
            W_FFT_ortho = CholeskyOrthConv(VFFT)
        elif self.mode == "cayley":
            W_FFT_ortho = cayley_func(VFFT)
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
        return W_FFT_ortho.reshape(
            -1,
            self.padding_input_size * (self.padding_input_size // 2 + 1),
            self.width,
            self.width,
        )

    def lipschitz(self):
        return 1.0

    def extra_repr(self) -> str:
        return f"{self.mode}: depth={self.depth}, width={self.width}, weight_rank={self.weight_rank}, quasi_residual={self.quasi_residual}, id_random_init={self.id_random_init}, zero_padding={self.zero_padding}, shift={self.shift}"

    def LOT_ortho(self, VFFT):
        sfft = VFFT @ VFFT.transpose(3, 4).conj()
        eps = sfft.diagonal(dim1=-2, dim2=-1).mean(dim=-1).mul(1e-5)
        sfft = sfft + eps.view(*eps.shape, 1, 1) * self.I_full
        norm_sfft = sfft.norm(p=None, dim=(3, 4), keepdim=True) + 1e-4
        sfft = sfft.div(norm_sfft)
        I = torch.eye(self.width, dtype=sfft.dtype).to(sfft.device).expand(sfft.shape)
        Y, Z = sfft, I
        for _ in range(10):
            T = (0.5 + 0j) * ((3 + 0j) * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        bfft = Z
        wfft_ortho = (bfft @ VFFT) / (norm_sfft.sqrt())
        return wfft_ortho

    def BRO(self, VFFT):
        assert VFFT.shape[-2] > VFFT.shape[-1]
        V_T_V_FFT = VFFT.transpose(3, 4).conj() @ VFFT
        eps = V_T_V_FFT.diagonal(dim1=-2, dim2=-1).mean(dim=-1).mul(1e-7)
        V_T_V_FFT = V_T_V_FFT + eps.view(*eps.shape, 1, 1) * self.I_low
        W_FFT_ortho = self.I_full - 2 * VFFT @ torch.linalg.solve(V_T_V_FFT, VFFT.transpose(3, 4).conj())
        return W_FFT_ortho

    def train(self, mode=True):
        self.training = mode
        if mode is False:
            with torch.no_grad():
                weights = self.get_wfft_ortho().detach()
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


class LiConvSOC(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        input_size: int,
        activation: nn.Module,
        centering: bool = True,
    ) -> None:
        super(LiConvSOC, self).__init__()
        self.depth = depth
        self.width = width
        self.blocks = nn.ModuleList([SOC(width, width, 3, bias=False) for _ in range(depth)])
        self.act = activation
        self.bias = nn.Parameter(torch.zeros(depth, width))
        if centering:
            running_mean = torch.zeros(depth, width)
            self.register_buffer("running_mean", running_mean)
        else:
            self.running_mean = 0
        self.centering = centering

    def forward(self, x):
        # for block in self.blocks:
        #     x = block(x)
        #     x = self.act(x)

        if self.centering and self.training:
            all_means = []
            for block, bias in zip(self.blocks, self.bias):
                z = block(x)
                mean = z.mean((0, 2, 3))
                all_means.append(mean.detach())
                z = z + (bias - mean).view(-1, 1, 1)
                x = self.act(z)
            all_means = torch.stack(all_means)
            self.running_mean += (all_means - self.running_mean) * 0.1
            return x

        biases = self.bias - self.running_mean
        for _, (block, bias) in enumerate(zip(self.blocks, biases)):
            z = block(x)
            z = z + bias.view(-1, 1, 1)
            x = self.act(z)
        return x

    def lipschitz(self):
        if self.training:
            return 1.0
        L = 1.0
        for block in self.blocks:
            lip = block.lipschitz()
            assert lip >= 1.0
            L *= lip
        return L

    def extra_repr(self) -> str:
        return f"depth={self.depth}, width={self.width}"


def cayley_func(W):
    if len(W.shape) == 2:
        return cayley_func(W[None])[0]

    shape = W.shape
    W = W.reshape(-1, shape[-2], shape[-1])
    _, cout, cin = W.shape
    if cin > cout:
        return cayley_func(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]

    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    tmp = torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)
    return tmp.reshape(shape)


class LiConvSLL(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        input_size: int,
    ) -> None:
        super(LiConvSLL, self).__init__()
        self.depth = depth
        self.width = width
        self.blocks = nn.ModuleList([SDPBasedLipschitzConvLayer(width, width, 3) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            # NOTE: SLL have built-in activation in the layer module for their formulation.
            x = block(x)
            # x = self.act(x)
        return x

    def lipschitz(self):
        return 1.0

    def extra_repr(self) -> str:
        return f"depth={self.depth}, width={self.width}"
