from typing import Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


def get_predictions(
    model: nn.Module, x: Tensor, eps_list: list[float], lc: Union[float, Tensor]
):
    if hasattr(model, "module"):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()

    clean_y = model(x)
    pred = clean_y.argmax(1)
    head_j = head[pred].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    perturbed_y_list = []
    for eps in eps_list:
        y_ = clean_y + lc * eps * head_ji
        y_ = y_.scatter(1, pred.view(-1, 1), -(10.0**10))
        y_ = y_.max(1)[0].reshape(-1, 1)
        y_ = torch.cat([clean_y, y_], dim=1)
        perturbed_y_list.append(y_)

    return clean_y, perturbed_y_list


def trades_loss(
    model: nn.Module,
    x: Tensor,
    label: Tensor,
    eps: float,
    lc: Union[float, Tensor],
    lambda_kl: float = 1.0,
    return_loss: bool = True,
    **kwargs
):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        lambda_kl (float): loss weight for the TRADES part.
        return_loss (bool): if True, compute and return the loss.
    """
    if hasattr(model, "module"):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)
    pred = y.argmax(1)
    head_j = head[pred].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    y_ = y + lc * eps * head_ji
    y_ = y_.scatter(1, pred.view(-1, 1), -(10.0**10))
    y_ = y_.max(1)[0].reshape(-1, 1)
    y_ = torch.cat([y, y_], dim=1)
    if return_loss:
        loss = nn.CrossEntropyLoss()(y, label)
        # If you are not clear why we compute the KL loss in this way,
        # please refer to https://github.com/hukkai/gloro_res/issues/2.
        KL_loss = y.log_softmax(dim=-1)[:, 0]
        KL_loss = KL_loss - y_.log_softmax(dim=-1)[:, 0]
        KL_loss = KL_loss.mean()
        loss = loss + KL_loss * lambda_kl
    else:
        loss = None
    return y, y_, loss


def margin_loss(
    model: nn.Module,
    x: Tensor,
    label: Tensor,
    eps: float,
    lc: Union[float, Tensor] = None,
    return_loss: bool = True,
    **kwargs
):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        return_loss (bool): if True, compute and return the loss.
    """
    if hasattr(model, "module"):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)

    head_j = head[label].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    margin = lc * head_ji
    y_ = y + eps * margin
    if return_loss:
        loss = nn.CrossEntropyLoss()(y_, label)
    else:
        loss = None
    return y, y_, loss


def logit_annealing_loss(
    model: nn.Module,
    x: Tensor,
    label: Tensor,
    eps: float,
    lc: Union[float, Tensor] = None,
    return_loss: bool = True,
    lip_reg: bool = True,
    **kwargs
):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        return_loss (bool): if True, compute and return the loss.
        lip_reg (bool): if True, incorporate EMMA for Lipschitz regularization.
    """
    if hasattr(model, "module"):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)
    loss = None
    y = y - y.gather(dim=1, index=label.reshape(-1, 1))
    head_j = head[label].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    margin = lc * head_ji

    eps_ji = -y / (margin * eps).clip(1e-8)
    eps_ji = eps_ji.clip(0.01, 1)  # .sqrt()

    y_ = y + eps_ji.detach() * eps * margin
    if return_loss:
        if lip_reg:
            loss = _logit_annealing_loss(
                num_classes=kwargs["num_classes"],
                offset=kwargs["offset"],
                temperature=kwargs["temperature"],
                gamma=kwargs["gamma"],
            )(y_, label)
        else:
            loss = _logit_annealing_loss(
                num_classes=kwargs["num_classes"],
                offset=kwargs["offset"],
                temperature=kwargs["temperature"],
                gamma=kwargs["gamma"],
            )(y, label)

    return y, y_, loss


def emma_loss(
    model: nn.Module,
    x: Tensor,
    label: Tensor,
    eps: float,
    lc: Union[float, Tensor] = None,
    return_loss: bool = True,
    **kwargs
):
    """
    Args:
        model (nn.Module): the trained model.
        x (Tensor): the input of the model.
        label (Tensor): the target of the model.
        eps (float): the robustness radius.
        lc (float or torch.Tensor): The lipschitz of the model backbone.
        return_loss (bool): if True, compute and return the loss.
    """
    if hasattr(model, "module"):
        head = model.module.head.get_weight()
    else:
        head = model.head.get_weight()
    y = model(x)
    loss = None
    y = y - y.gather(dim=1, index=label.reshape(-1, 1))
    head_j = head[label].unsqueeze(1)  # batch, 1, dim
    head_ji = head_j - head.unsqueeze(0)  # batch, num_class, dim
    head_ji = head_ji.norm(dim=-1)  # batch, num_class
    margin = lc * head_ji

    eps_ji = -y / (margin * eps).clip(1e-8)
    eps_ji = eps_ji.clip(0.01, 1)  # .sqrt()

    y_ = y + eps_ji.detach() * eps * margin
    if return_loss:
        loss = nn.CrossEntropyLoss()(y_, label)
    return y, y_, loss


class _logit_annealing_loss(nn.Module):
    def __init__(self, gamma=5.0, num_classes=None, offset=2.0, temperature=0.75):
        super(_logit_annealing_loss, self).__init__()

        self.gamma = gamma
        self.num_classes = num_classes
        self.offset = offset
        # self.offset = self.offset * math.sqrt(2)

        self.temperature = temperature

    def __call__(self, outputs, labels):
        if self.num_classes is None or self.offset is None or self.temperature is None:
            raise ValueError("num_classes, offset and temperature must be set.")

        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes)
        offset_outputs = outputs - self.offset * one_hot_labels
        offset_outputs /= self.temperature
        CE_loss = F.cross_entropy(offset_outputs, labels, reduction="none")
        labels = labels.view(-1, 1)
        pt = torch.exp(-CE_loss)
        loss = (1 - pt) ** self.gamma * CE_loss * self.temperature

        return loss.mean()
