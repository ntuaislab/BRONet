import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class Map2Vec(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        feat_size: int,
        out_dim: int,
        activation: nn.Module,
        conv_type: Union[str, Tuple[str, str]] = "bro",
        linear_type: str = "bro",
        input_size: int = 16,
        conv_patch_size: int = 4,
        conv_patch_size_2: int = 0,
    ) -> None:
        super(Map2Vec, self).__init__()
        # assert feat_size % 4 == 0
        if feat_size % conv_patch_size != 0:
            raise ValueError(
                f"`feat_size`: {feat_size} (feature map size) should be divisible by {conv_patch_size=}!"
            )

        self.conv_type_2 = None
        if type(conv_type) is tuple:
            conv_type, conv_type_2 = conv_type
            self.conv_type_2 = conv_type_2
        self.conv_type = conv_type
        if conv_patch_size_2 > 0 and self.conv_type_2 is None:
            self.conv_type_2 = self.conv_type

        self.linear_type = linear_type
        self.conv_patch_size = conv_patch_size
        self.conv_patch_size_2 = conv_patch_size_2
        mid_size = feat_size // conv_patch_size
        if conv_patch_size_2 > 0:
            mid_size = mid_size // conv_patch_size_2
        mid_dim = feat_dim * mid_size**2

        if conv_type != "l2":
            if conv_type == "bro":
                kernel = torch.randn(
                    feat_dim * conv_patch_size * conv_patch_size, feat_dim
                )
            else:
                kernel = torch.randn(
                    feat_dim, feat_dim * conv_patch_size * conv_patch_size
                )

            kernel = kernel / feat_dim**0.5 / conv_patch_size
            self.kernel = nn.Parameter(kernel)
        else:
            self.conv_layer = nn.LPPool2d(
                norm_type=2, kernel_size=conv_patch_size, stride=conv_patch_size
            )

        if self.conv_patch_size_2 > 0:
            assert (feat_size // conv_patch_size) % conv_patch_size_2 == 0
            if self.conv_type_2 != "l2":
                if self.conv_type_2 == "bro":
                    kernel_2 = torch.randn(
                        feat_dim * conv_patch_size_2 * conv_patch_size_2, feat_dim
                    )
                else:
                    kernel_2 = torch.randn(
                        feat_dim, feat_dim * conv_patch_size_2 * conv_patch_size_2
                    )
                kernel_2 = kernel_2 / feat_dim**0.5 / conv_patch_size_2
                self.kernel_2 = nn.Parameter(kernel_2)
            else:
                self.conv_layer_2 = nn.LPPool2d(
                    norm_type=2, kernel_size=conv_patch_size_2, stride=conv_patch_size_2
                )

        if linear_type == "bro":
            weight = torch.randn(mid_dim, out_dim) / mid_dim**0.5
        else:
            weight = torch.randn(out_dim, mid_dim) / mid_dim**0.5
        self.linear_out_dim = out_dim
        self.linear_in_dim = mid_dim
        self.weight = nn.Parameter(weight)

        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.feat_dim = feat_dim
        self.mid_size = mid_size
        self.activation = activation
        self.input_size = input_size

    @staticmethod
    def get_cholesky_weight(weight):
        Sigma = weight @ weight.T
        eps = Sigma.diag().mean().div(1000.0).item()
        Sigma = Sigma + eps * torch.eye(
            Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype
        )
        L = torch.linalg.cholesky(Sigma)
        weight = torch.linalg.solve_triangular(L, weight, upper=False)
        return weight

    @staticmethod
    def get_bro_weight(weight, out_dim, in_dim):
        assert weight.shape[0] > weight.shape[1]
        Sigma = weight.T @ weight
        eps = Sigma.diag().mean().mul(1e-5).item()
        Sigma = Sigma + eps * torch.eye(
            Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype
        )
        weight = torch.eye(
            weight.shape[0], device=weight.device, dtype=weight.dtype
        ) - 2 * weight @ torch.linalg.solve(Sigma, weight.T)
        weight = weight[:out_dim, :in_dim]
        return weight

    def get_weight(self):
        kernel_, kernel_2 = None, None
        if self.conv_type == "cholesky":
            kernel_ = self.get_cholesky_weight(self.kernel)
            kernel_ = kernel_.reshape(
                self.feat_dim, self.feat_dim, self.conv_patch_size, self.conv_patch_size
            )
        elif self.conv_type == "bro":
            kernel_ = self.get_bro_weight(
                self.kernel, self.feat_dim, self.feat_dim * self.conv_patch_size**2
            )
            kernel_ = kernel_.reshape(
                self.feat_dim, self.feat_dim, self.conv_patch_size, self.conv_patch_size
            )
        elif self.conv_type != "l2":
            raise ValueError("Unsupported `conv` type!")
        if self.conv_patch_size_2 > 0:
            if self.conv_type_2 == "cholesky":
                kernel_2 = self.get_cholesky_weight(self.kernel_2)
                kernel_2 = kernel_2.reshape(
                    self.feat_dim,
                    self.feat_dim,
                    self.conv_patch_size_2,
                    self.conv_patch_size_2,
                )
            elif self.conv_type_2 == "bro":
                kernel_2 = self.get_bro_weight(
                    self.kernel_2,
                    self.feat_dim,
                    self.feat_dim * self.conv_patch_size_2**2,
                )
                kernel_2 = kernel_2.reshape(
                    self.feat_dim,
                    self.feat_dim,
                    self.conv_patch_size_2,
                    self.conv_patch_size_2,
                )
            elif self.conv_type != "l2":
                raise ValueError("Unsupported `conv` type!")
        if self.linear_type == "cholesky":
            weight_ = self.get_cholesky_weight(self.weight)
        elif self.linear_type == "bro":
            weight_ = self.get_bro_weight(
                self.weight, self.linear_out_dim, self.linear_in_dim
            )
        else:
            raise ValueError("Unsupported `linear` type!")

        return kernel_, kernel_2, weight_

    def forward(self, x):
        kernel, kernel_2, weight = self.get_weight()
        if self.conv_type == "l2":
            x = self.conv_layer(x)
        elif kernel is not None:
            x = F.conv2d(x, kernel, stride=self.conv_patch_size)
        if self.conv_type_2 == "l2":
            x = self.conv_layer_2(x)
        elif kernel_2 is not None:
            x = F.conv2d(x, kernel_2, stride=self.conv_patch_size_2)
        x = x.reshape(x.shape[0], -1)
        x = F.linear(x, weight, self.bias)
        x = self.activation(x)
        return x

    def lipschitz(self):
        if self.training:
            return 1.0

        lc = 1.0
        kernel, kernel_2, weight = self.get_weight()
        if kernel is not None:
            kernel = kernel.reshape(self.feat_dim, -1)
            lc = kernel.svd().S.max()
        if kernel_2 is not None:
            kernel_2 = kernel_2.reshape(self.feat_dim, -1)
            lc = lc * kernel_2.svd().S.max()

        lc = lc * weight.svd().S.max()
        return lc.item()

    def extra_repr(self) -> str:
        if self.conv_type_2 is not None:
            return f"conv_type/conv_type_2={self.conv_type}/{self.conv_type_2}, channel={self.feat_dim}, stride/patch_size/kernel_size = {self.conv_patch_size}, stride/patch_size/kernel_size_2 = {self.conv_patch_size_2}, linear_type={self.linear_type} in={self.linear_in_dim} out={self.linear_out_dim}."
        return f"conv_type={self.conv_type}, channel={self.feat_dim}, stride/patch_size/kernel_size = {self.conv_patch_size}, linear_type={self.linear_type} in={self.linear_in_dim} out={self.linear_out_dim}."
