from typing import Optional
import torch
import torch.nn as nn

from .dense import LiBroMLP, LiResMLP
from .backbone import LiResConvOrtho, LiConvSOC, LiResConv, LiConvSLL
from .layers import build_activation
from .stem import build_stem
from .neck import Map2Vec
from .head import head
from .orthoconv_v2 import OrthoConvV2


class BRONet(nn.Module):
    def __init__(
        self,
        depth: int = 12,
        width: int = 256,
        input_size: int = 32,
        num_classes: int = 10,
        num_lc_iter: int = 10,
        act_name: str = "MinMax",
        use_lln: bool = True,
        **kwargs
    ):
        super(BRONet, self).__init__()

        backbone_type = kwargs.get("backbone_type", "default").lower()
        neck_conv_type = kwargs.get("neck_conv_type", "default").lower()
        neck_linear_type = kwargs.get("neck_linear_type", "default").lower()
        self.depth = depth

        stem_kernel_size = kwargs.get("stem_kernel_size", 5)
        stem, feature_size = build_stem(input_size, width, act_name, stem_kernel_size)

        self.stem = stem
        kwargs["input_size"] = feature_size

        if act_name == "hh1" or act_name == "hh2":
            raise NotImplementedError(
                "Should initiate new activation modules for each block if the activation is trainable!"
            )
        activation = build_activation(act_name, dim=1, channels=width)
        self.conv_1 = None
        if "hybrid" in backbone_type:
            depth_1 = kwargs.get("depth_1", depth // 2)
            depth_2 = kwargs.get("depth_2", depth - depth_1)
            assert depth_1 + depth_2 == depth
            self.conv_1 = LiResConv(
                width=width,
                depth=depth_1,
                input_size=feature_size,
                activation=activation,
            )

            backbone_type = backbone_type.replace("hybrid", "")
            depth = depth_2

        if backbone_type == "soc":
            self.conv = LiConvSOC(
                width=width,
                depth=depth,
                input_size=feature_size,
                activation=activation,
            )
        elif backbone_type == "nonortho":
            self.conv = LiResConv(
                width=width,
                depth=depth,
                input_size=feature_size,
                activation=activation,
            )
        elif backbone_type == "sll":
            self.conv = LiConvSLL(
                width=width,
                depth=depth,
                input_size=feature_size,
                # activation=activation,
            )
        else:
            if "v2" in backbone_type:
                backbone_type = backbone_type.replace("v2", "")
                self.conv = OrthoConvV2(
                    width=width,
                    depth=depth,
                    input_size=feature_size,
                    activation=activation,
                    mode=backbone_type,
                    weight_rank_ratio=kwargs.get("backbone_weight_rank_ratio", 0.5),
                    quasi_residual=kwargs.get("backbone_quasi_residual", True),
                    id_random_init=kwargs.get("backbone_id_random_init", False),
                    zero_padding=kwargs.get("backbone_zero_padding", True),
                )
                feature_size = self.conv.padding_input_sizes[-1]
            else:
                self.conv = LiResConvOrtho(
                    width=width,
                    depth=depth,
                    input_size=feature_size,
                    activation=activation,
                    mode=backbone_type,
                    centering=kwargs.get("backbone_centering", True),
                    weight_rank_ratio=kwargs.get("backbone_weight_rank_ratio", 0.25),
                    quasi_residual=kwargs.get("backbone_quasi_residual", True),
                    id_random_init=kwargs.get("backbone_id_random_init", False),
                    zero_padding=kwargs.get("backbone_zero_padding", True),
                )

        out_dim = kwargs.get("dense_width", 2048)

        self.neck = Map2Vec(
            feat_dim=width,
            feat_size=feature_size,
            out_dim=out_dim,
            activation=activation,
            conv_type=neck_conv_type,
            linear_type=neck_linear_type,
            input_size=feature_size,
            conv_patch_size=kwargs.get("neck_conv_patch_size", 4),
            conv_patch_size_2=kwargs.get("neck_conv_patch_size_2", 0),
        )

        self.linear_num = kwargs.get("linear_num", 8)
        if self.linear_num > 0:
            dense_weight_rank_ratio = kwargs.get("dense_weight_rank_ratio", 0.5)
            self.linear_type = kwargs.get("dense_type", "bro").lower()
            if self.linear_type == "bro":
                self.linear = LiBroMLP(
                    num_features=out_dim,
                    depth=self.linear_num,
                    activation=activation,
                    weight_rank_ratio=dense_weight_rank_ratio,
                    id_random_init=kwargs.get("dense_id_random_init", False),
                    quasi_residual=kwargs.get("dense_quasi_residual", True),
                )
            elif self.linear_type == "cholesky":
                self.linear = LiResMLP(
                    num_features=out_dim,
                    depth=self.linear_num,
                    activation=activation,
                )
            else:
                raise ValueError("Unknown linear type!")

        self.head = head(out_dim, num_classes, use_lln)

        self.num_lc_iter = num_lc_iter
        self.set_num_lc_iter()

    def set_num_lc_iter(self, num_lc_iter: Optional[int] = None) -> None:
        if num_lc_iter is None:
            num_lc_iter = self.num_lc_iter
        for m in self.modules():
            setattr(m, "num_lc_iter", num_lc_iter)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input image tensors in [0, 1]
            return_feat (bool): if true, only return the extracted features.

        """
        x = x.sub(0.5)
        x = self.stem(x)
        if self.conv_1 is not None:
            x = self.conv_1(x)
        x = self.conv(x)
        x = self.neck(x)
        if hasattr(self, "linear"):
            x = self.linear(x)
        if return_feat:
            return x
        x = self.head(x)
        return x

    def sub_lipschitz(self) -> torch.Tensor:
        """Compute the lipschitz constant of the model except the head."""
        lc = self.stem.lipschitz()
        lc = lc * self.neck.lipschitz()
        if self.conv is not None:
            lc = lc * self.conv.lipschitz()
        if self.conv_1 is not None:
            lc = lc * self.conv_1.lipschitz()
        if hasattr(self, "linear"):
            lc = lc * self.linear.lipschitz()
        return lc
