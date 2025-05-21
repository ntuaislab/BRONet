import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn as dist_nn
from .cholesky_grad import CholeskyOrth


class LiResMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        depth: int,
        activation: nn.Module,
        num_lc_iter: int = 10,
    ) -> None:
        super(LiResMLP, self).__init__()
        weights = torch.randn(depth, num_features, num_features)
        weights = weights / num_features
        self.weights = nn.Parameter(weights)

        self.gamma = nn.Parameter(torch.ones(depth, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(depth, num_features))

        self.register_buffer("identity", torch.eye(num_features))
        self.register_buffer("eval_weight", torch.zeros_like(weights))

        self.act = activation

        self.scale = depth**-0.5
        self.num_lc_iter = num_lc_iter
        self.depth = depth
        self.width = num_features

        flag = torch.distributed.is_initialized()
        self.flag = flag and depth % torch.distributed.get_world_size() == 1
        if torch.distributed.get_world_size() < 8:
            # distributed is slower with less GPUs
            self.flag = False
        # self.flag = False

    def get_weight(self) -> torch.Tensor:
        if self.flag:
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
            num_per_gpu = self.depth // world
            index = range(rank * num_per_gpu, (rank + 1) * num_per_gpu)
            _W = (
                self.identity.data
                + self.weights[index] * self.gamma[index] * self.scale
            )
            _W = CholeskyOrth(_W).contiguous()
            W = dist_nn.functional.all_gather(_W)
            W = torch.cat(W, dim=0)
            return W
        W = self.identity.data + self.weights * self.gamma * self.scale
        W = CholeskyOrth(W)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weights = self.get_weight()
        else:
            weights = self.eval_weight

        for weight, bias in zip(weights, self.bias):
            x = F.linear(x, weight, bias)
            x = self.act(x)
        return x

    def lipschitz(self):
        if self.training:
            return 1.0

        weights = self.get_weight()
        return torch.linalg.matrix_norm(weights, ord=2).prod()

    def train(self, mode=True):
        self.training = mode
        if mode is False:
            weights = self.get_weight().detach()
            self.eval_weight += weights - self.eval_weight
        # else:
        #     if hasattr(self, "eval_weight"):
        #         del self.eval_weight
        return self

    def extra_repr(self) -> str:
        return f"depth={self.depth}, width={self.width}"


class LiBroMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        depth: int,
        activation: nn.Module,
        weight_rank_ratio: float = 0.25,
        id_random_init: bool = False,
        quasi_residual: bool = True,
    ) -> None:
        super(LiBroMLP, self).__init__()
        weight_rank = self.weight_rank_mapping(num_features, weight_rank_ratio)
        self.weight_rank = weight_rank
        self.id_random_init = id_random_init
        self.quasi_residual = quasi_residual

        self.act = activation
        self.depth = depth
        self.width = num_features
        self.scale = depth**-0.5

        residual_identity = torch.zeros(num_features, weight_rank)
        residual_identity[:weight_rank, :weight_rank] = torch.eye(weight_rank)
        self.register_buffer("residual_identity", residual_identity)

        weights = torch.randn(depth, num_features, weight_rank)
        weights = weights / num_features

        if self.id_random_init:
            assert not self.quasi_residual
            weights = self.scale * weights + self.residual_identity

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.zeros(depth, num_features))

        self.register_buffer("identity", torch.eye(num_features))
        self.register_buffer("identity_half", torch.eye(weight_rank))
        # self.register_buffer(
        #     "eval_weights", torch.zeros(depth, num_features, num_features)
        # )

        self.gamma = nn.Parameter(torch.ones(depth, num_features, 1))

        flag = torch.distributed.is_initialized()
        self.flag = flag and depth % torch.distributed.get_world_size() == 0
        if torch.distributed.get_world_size() < 8:
            # distributed is slower with less GPUs
            self.flag = False

    def get_weights(self):
        if self.flag:
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
            num_per_gpu = self.depth // world
            index = range(rank * num_per_gpu, (rank + 1) * num_per_gpu)
            if self.quasi_residual:
                _V = (
                    self.residual_identity.data
                    + self.weights[index] * self.gamma[index] * self.scale
                )
            else:
                _V = self.weights[index] * self.gamma[index]
            Sigma = _V.transpose(1, 2) @ _V
            eps = Sigma.diagonal(dim1=1, dim2=2).mean(-1).mul(1e-7)
            Sigma = Sigma + eps.view(*eps.shape, 1, 1) * self.identity_half.unsqueeze(0)
            _W = self.identity - 2 * _V @ torch.linalg.solve(Sigma, _V.transpose(1, 2))
            _W = _W.contiguous()
            W = dist_nn.functional.all_gather(_W)
            W = torch.cat(W, dim=0)
            return W
        V = self.residual_identity + self.weights * self.gamma * self.scale
        Sigma = V.transpose(1, 2) @ V
        eps = Sigma.diagonal(dim1=1, dim2=2).mean(-1).mul(1e-7)
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
            with torch.no_grad():
                weights = self.get_weights().detach()
                # self.eval_weights += weights - self.eval_weights
                self.eval_weights = weights
        else:
            if hasattr(self, "eval_weights"):
                del self.eval_weights
        return self

    def lipschitz(self):
        return 1.0

    def extra_repr(self) -> str:
        return f"BRO: depth={self.depth}, weight_rank/width = {self.weight_rank}/{self.width}, distributed={self.flag}, quasi_residual={self.quasi_residual}, id_random_init={self.id_random_init}"

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
