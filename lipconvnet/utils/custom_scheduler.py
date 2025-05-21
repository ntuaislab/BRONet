import math
import numpy as np


class CustomCosineScheduler(object):
    def __init__(self, optimizer, iter_per_epoch, max_epoch, warmup_epoch=10):

        self.warmup_iters = iter_per_epoch * warmup_epoch
        self.current_iter = 0
        self.num_iters = iter_per_epoch * (max_epoch - warmup_epoch)
        self.base_lr = 0.0
        self.optimizer = optimizer

    def step(self):
        if self.current_iter == 0:
            for group in self.optimizer.param_groups:
                lr = group["lr"]
                group.setdefault("initial_lr", group["lr"])
                self.base_lr = max(self.base_lr, lr)

        self.current_iter += 1
        if self.current_iter < self.warmup_iters:
            lr_ratio = self.current_iter / self.warmup_iters
        else:
            lr_ratio = self._cosine_get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_ratio * param_group["initial_lr"]
        self.last_lr = lr_ratio * self.base_lr
        return self.last_lr

    def get_last_lr(self):
        return self.last_lr

    def _cosine_get_lr(self):
        process = (self.current_iter - self.warmup_iters) / self.num_iters
        lr_ratio = 0.5 * (1 + math.cos(process * math.pi))
        return max(lr_ratio, 1e-5)

class TriangularLRScheduler:
    def __init__(self, optimizer, lr_steps, lr):
        self.optimizer = optimizer
        self.epochs = lr_steps
        self.lr = lr

    def step(self, t):
        lr = np.interp(
            [t],
            [0, self.epochs * 2 // 5, self.epochs * 4 // 5, self.epochs],
            [0, self.lr, self.lr / 20.0, 0],
        )[0]
        self.optimizer.param_groups[0].update(lr=lr)

    def get_last_lr(self):
        return self.lr
