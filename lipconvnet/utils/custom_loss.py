import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assuming inputs are raw logits from the last layer of your network
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Convert targets to same shape as inputs for broadcasting
        targets = targets.view(-1, 1)
        pt = torch.exp(-BCE_loss)  # Probabilities of the ground truth class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class LossXent(nn.Module):

    def __init__(self, n_classes, offset, temperature):
        super(LossXent, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.offset = offset
        self.temperature = temperature

    def __call__(self, outputs, labels):
        one_hot_labels = F.one_hot(labels, num_classes=self.n_classes)
        offset_outputs = outputs - self.offset * one_hot_labels
        offset_outputs /= self.temperature
        loss = self.criterion(offset_outputs, labels) * self.temperature
        return loss


class LossLogitAnnealing(nn.Module):

    def __init__(self, n_classes, offset, temperature, la_alpha=0.25, la_beta=2.0):
        super().__init__()
        self.criterion = FocalLoss(gamma=la_beta, alpha=la_alpha)
        self.n_classes = n_classes
        self.offset = offset
        self.temperature = temperature

        self.la_alpha = la_alpha
        self.la_beta = la_beta

    def __call__(self, outputs, labels):
        one_hot_labels = F.one_hot(labels, num_classes=self.n_classes)
        offset_outputs = outputs - self.offset * one_hot_labels
        offset_outputs /= self.temperature
        loss = self.criterion(offset_outputs, labels) * self.temperature
        return loss
