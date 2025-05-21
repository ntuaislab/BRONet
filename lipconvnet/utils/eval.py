import torch
import torch.nn.functional as F
import numpy as np
import math

from . import lower_limit, upper_limit, std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(
                d,
                lower_limit - X[index[0], :, :, :],
                upper_limit - X[index[0], :, :, :],
            )
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction="none").detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, limit_n=float("inf")):
    epsilon = (8 / 255.0) / std
    alpha = (2 / 255.0) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for _, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss / n, pgd_acc / n


def attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    epsilon_value = epsilon[0].item()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            # d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = d + alpha * torch.sign(g) / torch.norm(g, p=2, dim=(1, 2, 3), keepdim=True)
            # d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            d_norm = torch.norm(d, p=2, dim=(1, 2, 3))
            for d_norm_idx in range(d_norm.shape[0]):
                if d_norm[d_norm_idx].item() > epsilon_value:
                    d[d_norm_idx] = d[d_norm_idx] * epsilon_value / d_norm[d_norm_idx]
            # if d_norm > epsilon:
            #     d = d * epsilon / d_norm
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction="none").detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd_l2(test_loader, model, attack_iters, restarts, limit_n=float("inf")):
    epsilon = (36 / 255.0) / std
    alpha = epsilon / 5.0
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for _, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss / n, pgd_acc / n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


# TODO: import the following four functions in train_robust.py
def ortho_certificates(output, class_indices, L):
    # DONE: almost 1-Lipschitz activation certification
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)

    onehot = torch.zeros_like(output).cuda()
    onehot[torch.arange(output.shape[0]), class_indices] = 1.0
    output_trunc = output - onehot * 1e6

    output_class_indices = output[batch_indices, class_indices]
    output_nextmax = torch.max(output_trunc, dim=1)[0]
    output_diff = output_class_indices - output_nextmax
    # return output_diff/(math.sqrt(2)*L*torch.pow(torch.tensor(lip_const), torch.tensor(lip_factor)))
    return output_diff / (math.sqrt(2) * L)


def lln_certificates(output, class_indices, last_layer, L):
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)

    onehot = torch.zeros_like(output).cuda()
    onehot[batch_indices, class_indices] = 1.0
    output_trunc = output - onehot * 1e6

    lln_weight = last_layer.lln_weight
    lln_weight_indices = lln_weight[class_indices, :]
    lln_weight_diff = lln_weight_indices.unsqueeze(1) - lln_weight.unsqueeze(0)
    lln_weight_diff_norm = torch.norm(lln_weight_diff, dim=2)
    lln_weight_diff_norm = lln_weight_diff_norm + onehot

    output_class_indices = output[batch_indices, class_indices]
    output_diff = output_class_indices.unsqueeze(1) - output_trunc
    all_certificates = output_diff / (lln_weight_diff_norm * L)
    return torch.min(all_certificates, dim=1)[0]


def evaluate_certificates(test_loader, model, L, epsilon=36.0, gt=False):
    losses_list = []
    certificates_list = []
    correct_list = []
    model.eval()

    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y, reduction="none")
            losses_list.append(loss)

            output_max, output_amax = torch.max(output, dim=1)
            correct = output_amax == y

            if gt:
                output_amax = y
            model_lipschitz = model.set_and_get_lipschitz_constant()
            L *= model_lipschitz

            if model.lln:
                certificates = lln_certificates(output, output_amax, model.last_layer, L)
            else:
                certificates = ortho_certificates(output, output_amax, L)

            certificates_list.append(certificates)
            correct_list.append(correct)

        losses_array = torch.cat(losses_list, dim=0).cpu().numpy()
        certificates_array = torch.cat(certificates_list, dim=0).cpu().numpy()
        correct_array = torch.cat(correct_list, dim=0).cpu().numpy()
    return losses_array, correct_array, certificates_array


def robust_statistics(losses_arr, correct_arr, certificates_arr, epsilon_list=[36.0, 72.0, 108.0, 144.0, 180.0, 216.0], truncated_theshold=255.0):
    mean_loss = np.mean(losses_arr)
    mean_acc = np.mean(correct_arr)
    # mean_certs = (certificates_arr * correct_arr).sum() / correct_arr.sum()
    # _mean_certs = (certificates_arr * correct_arr).mean()
    trimmed_acr = (np.clip(certificates_arr, a_min=0, a_max=truncated_theshold / 255.0) * correct_arr).mean()
    # print(f"mean_certs: {_mean_certs}/{mean_certs}, truncated_acr: {truncated_acr},\n\n ")

    trimmed_acr = round(trimmed_acr, 4)

    robust_acc_list = []
    for epsilon in epsilon_list:
        robust_correct_arr = (certificates_arr > (epsilon / 255.0)) & correct_arr
        robust_acc = robust_correct_arr.sum() / robust_correct_arr.shape[0]
        robust_acc_list.append(robust_acc)
    return mean_loss, mean_acc, trimmed_acr, robust_acc_list
