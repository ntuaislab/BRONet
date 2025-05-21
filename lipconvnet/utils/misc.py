import glob
import math
import time
import datetime
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


from models.layers import (
    Cayley,
    CayleyLinear,
    SOC,
    BRO,
    BROLinear,
    LOT,
)

from .custom_scheduler import CustomCosineScheduler, TriangularLRScheduler

from .custom_activations import (
    MaxMin,
    VReLU,
    HouseHolder,
    HouseHolder_Order_2,
)

from .custom_loss import (
    LossXent,
    LossLogitAnnealing,
)


conv_mapping = {
    "standard": nn.Conv2d,
    "soc": SOC,
    "cayley": Cayley,
    "bro": BRO,
    "lot": LOT,
}


def activation_mapping(activation_name, channels=None):
    # Dictionary to hold all activations
    activation_dict = {
        "hh1": lambda: HouseHolder(channels=channels),
        "hh2": lambda: HouseHolder_Order_2(channels=channels),
        "vrelu": lambda: VReLU(),
        "relu": lambda: nn.ReLU(),
        "sigmoid": lambda: nn.Sigmoid(),
        "tanh": lambda: nn.Tanh(),
        "swish": lambda: F.silu,
        "softplus": lambda: F.softplus,
        "maxmin": lambda: MaxMin(),
        "rrelu": lambda: F.rrelu,
        "leakyrelu": lambda: F.leaky_relu,
    }

    # Fetch the activation function from the dictionary
    if activation_name in activation_dict:
        return activation_dict[activation_name]()
    else:
        raise ValueError(f"Activation function '{activation_name}' is not defined.")


def dense_mapping(dense_name):
    if dense_name == "standard":
        dense_layer = nn.Linear
    elif dense_name == "cayley":
        dense_layer = CayleyLinear
    elif dense_name == "bro":
        dense_layer = BROLinear
    else:
        raise ValueError(f"Dense Layer {dense_name} not supported")
    return dense_layer


def loss_mapping(loss_name, args):
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "xent":
        criterion = LossXent(
            n_classes=args.num_classes,
            temperature=args.temperature,
            offset=args.offset * math.sqrt(2),
        )
    elif args.loss == "la":
        criterion = LossLogitAnnealing(
            n_classes=args.num_classes,
            temperature=args.temperature,
            offset=args.offset * math.sqrt(2),
            la_alpha=args.la_alpha,
            la_beta=args.la_beta,
        )  # temperature=0.25, offset=math.sqrt(2)
    else:
        raise ValueError(f"Loss {loss_name} not supported")
    return criterion


def lr_scheduler_mapping(lr_scheduler_name, opt, lr_steps, lr, max_epoch, custom_warmup_epoch):
    if lr_scheduler_name == "default":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1)
    elif lr_scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=lr_steps, eta_min=0)
    elif lr_scheduler_name == "cosine-wr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=lr_steps, T_mult=1, eta_min=0)
    elif lr_scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.1,
            patience=8,
        )
    elif lr_scheduler_name == "triangle":
        scheduler = TriangularLRScheduler(opt, lr_steps=lr_steps, lr=lr)
    elif lr_scheduler_name == "custom_cosine":
        iter_per_epoch = lr_steps // max_epoch
        scheduler = CustomCosineScheduler(opt, iter_per_epoch, max_epoch, warmup_epoch=custom_warmup_epoch)

    else:
        raise ValueError(f"LR Scheduler {lr_scheduler_name} not supported")
    return scheduler


def cr_scheduler_mapping(cr_scheduler_name, loss, current_epoch, epoch, curr_cert, gamma):
    if gamma <= 0:
        return loss
    else:
        if cr_scheduler_name == "default":
            loss = loss - gamma * F.relu(curr_cert).mean()
        elif cr_scheduler_name == "linear":
            loss = loss - gamma * (epoch / epoch) * F.relu(curr_cert).mean()
        elif cr_scheduler_name == "quad":
            loss = loss - gamma * (epoch / epoch) ** 2 * F.relu(curr_cert).mean()
        elif cr_scheduler_name == "sqrt":
            loss = loss - gamma * math.sqrt(epoch / epoch) * F.relu(curr_cert).mean()
        elif cr_scheduler_name == "cosine":
            loss = loss - gamma * math.cos(epoch / epoch * math.pi / 2 - math.pi / 2) * F.relu(curr_cert).mean()
        elif cr_scheduler_name == "cosine-alt":
            loss = loss - gamma * (1 / 2) * (1 - math.cos(epoch / epoch * math.pi)) * F.relu(curr_cert).mean()
        return loss


def get_parameter_lists(model):
    conv_params = []
    activation_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "activation" in name:
                activation_params.append(param)
            elif "conv" in name:
                conv_params.append(param)
            else:
                other_params.append(param)
    return conv_params, activation_params, other_params


def increment_path(path, exist_ok=False, sep=""):
    # INFO: Increment path (os-agnostic), i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [int(d.split(sep)[-1]) for d in dirs if d.split(sep)[-1].isnumeric()]
        n = max(matches) + 1 if matches else 2  # increment number
        print(f"Path already exists, replace with {path}{sep}{n}")
        return f"{path}{sep}{n}"  # update path


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()


def get_git_commit_msg():
    return subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("ascii").strip()


def get_git_timestamp():
    return subprocess.check_output(["git", "log", "-1", "--pretty=%cd", "--date=format:%Y-%m-%d %H:%M:%S"]).decode("ascii").strip()


class Timer:
    def __init__(self, total_epochs, moving_average_window=5):
        self.total_epochs = total_epochs
        self.moving_average_window = moving_average_window
        self.start_time = time.time()
        self.original_start_time = time.time()
        self.epoch_times = []

    def update(self, current_epoch):
        current_time = time.time()
        epoch_duration = current_time - self.start_time
        self.epoch_times.append(epoch_duration)

        if len(self.epoch_times) > self.moving_average_window:
            self.epoch_times.pop(0)
        self.start_time = current_time
        past_time = current_time - self.original_start_time
        return self.estimate_remaining_time(current_epoch), past_time

    def estimate_remaining_time(self, current_epoch):
        average_time_per_epoch = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (current_epoch + 1)
        return average_time_per_epoch * remaining_epochs

    def remaining_time(self, current_epoch):
        estimated_time, past_time = self.update(current_epoch)
        return (
            f"{datetime.timedelta(seconds=round(estimated_time))}",
            f"{datetime.timedelta(seconds=round(past_time))}",
        )
