import os
import torch
import numpy as np
from tqdm import tqdm
from autoattack import AutoAttack
import matplotlib.pyplot as plt
import argparse
import json
import yaml
from dataclasses import dataclass
from typing import List


from utils.data import cifar10_mean, cifar10_std, get_loaders, ModelNormWrapper
from utils.eval import evaluate_certificates, robust_statistics, evaluate_pgd, evaluate_pgd_l2

from .base import BaseTrainer


class Evaluator(BaseTrainer):
    def __init__(self, config):
        super(Evaluator, self).__init__(config)

    @staticmethod
    def model_summary(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = round(sum([np.prod(p.size()) for p in model_parameters]) / 1_000_000)
        print(f"Model has {params}M trainable parameters")
        return params

    @staticmethod
    def plot_histograms(train_loader, test_loader, model_test, L):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        losses_arr, correct_arr, certificates_arr = evaluate_certificates(train_loader, model_test, L, gt=True)
        plt.subplot(1, 2, 1)
        plt.hist(certificates_arr, bins=list(np.linspace(-2, 2, 60)), alpha=0.9, color='red', label='train', histtype="step")
        plt.grid()
        plt.xlabel('Margin', fontsize=15)
        plt.ylabel('number', fontsize=15)
        plt.legend(fontsize=15)

        losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model_test, L, gt=True)
        plt.subplot(1, 2, 2)
        plt.hist(certificates_arr, bins=list(np.linspace(-2, 2, 60)), alpha=0.9, color='red', label='test', histtype="step")
        plt.grid()
        plt.xlabel('Margin', fontsize=15)
        plt.legend(fontsize=15)

        plt.tight_layout()
        plt.savefig('histogram.png')
        plt.close()

    def __call__(self):
        if self.config.checkpoint.endswith('pth'):
            raise ValueError('--checkpoint, please provide a directory, not a file')

        best_model_path = os.path.join(self.config.checkpoint, 'best.pth')
        print("Evaluation on checkpoint: ", best_model_path)

        if self.config.dataset == "cifar10":
            self.config.num_classes = 10
            means = cifar10_mean
            stds = cifar10_std
        elif self.config.dataset == "cifar100":
            self.config.num_classes = 100
            means = cifar10_mean
            stds = cifar10_std
        elif self.config.dataset == "tinyimg":
            self.config.num_classes = 200
            # HACK: temporary use cifar10 stats for tinyimgnet
            means = cifar10_mean
            stds = cifar10_std
        else:
            raise Exception("Unknown dataset")

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        _, test_loader = get_loaders(
            self.config.data_dir,
            self.config.batch_size,
            self.config.dataset,
        )
        std = cifar10_std

        std = torch.tensor(std).cuda()
        L = 1 / torch.max(std)

        model_test = self.init_model().cuda()
        model_test.load_state_dict(torch.load(best_model_path))
        model_test.float()
        model_test.eval()

        if self.config.mode == 'certified':
            losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model_test, L)
            test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(losses_arr, correct_arr, certificates_arr)
            print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')
            print(f'Clean/36/72/108 : {test_acc}/{test_robust_acc_list[0]}/{test_robust_acc_list[1]}/{test_robust_acc_list[2]}')

            self.model_summary(model_test)

        if self.config.mode == 'pgd-atk':
            print('=== Evaluate PGD ===')
            pgd_l2_loss, pgd_l2_acc = evaluate_pgd_l2(test_loader, model_test, attack_iters=5, restarts=2, limit_n=float("inf"))
            print(f"{pgd_l2_loss=}, {pgd_l2_acc=}")
            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, attack_iters=5, restarts=2, limit_n=float("inf"))
            print(f"{pgd_loss=}, {pgd_acc=}")

        if self.config.mode == 'auto-atk':
            print('=== Evaluate AutoAttack ===')
            print(f'Using attack budget: {self.config.epsilon}')
            _, test_loader = get_loaders(self.config.data_dir, self.config.batch_size, self.config.dataset, normalize=False)
            normalized_model = ModelNormWrapper(model_test, means, stds)
            images, labels = [], []
            for image, label in test_loader:
                images.append(image)
                labels.append(label)
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)

            adversary = AutoAttack(
                normalized_model, norm="L2", eps=self.config.epsilon / 255.0, version="standard", log_path=os.path.join(self.config.checkpoint, f"autoattack{self.config.epsilon}.log")
            )
            _ = adversary.run_standard_evaluation(images, labels, bs=self.config.batch_size)

        if self.config.mode == 'lower-lipschitz':
            Lip = []
            for _ in range(100):
                index1 = np.random.randint(0, len(test_loader.dataset))
                index2 = np.random.randint(0, len(test_loader.dataset))
                X1, y1 = test_loader.dataset[index1]
                X2, y2 = test_loader.dataset[index2]

                output1 = model_test(X1.unsqueeze(0).cuda())
                output2 = model_test(X2.unsqueeze(0).cuda())

                inputData = X1 - X2
                outputData = output1 - output2
                print(_, ':', outputData.norm().item() / inputData.norm().item(), end='  \r')
                Lip.append(outputData.norm().item() / inputData.norm().item())
            print('-' * 20)
            Lip = np.array(Lip)
            print('Max:', Lip.max())
            print('Min:', Lip.min())
            print('Mean:', Lip.mean())
            print('Std:', Lip.std())

        if self.config.mode == 'upper-lipschitz':
            inputs = torch.rand(32, 3, 32, 32) - 0.5

            inputs = inputs.cuda()
            inputs.requires_grad = True

            optimizer = torch.optim.Adam([inputs], lr=1e-4)
            lc = []
            with tqdm(range(100000), desc="Initial Progress") as pbar:
                for k in pbar:
                    optimizer.zero_grad()
                    outputs = model_test(inputs)
                    if k == 0:
                        L = model_test.set_and_get_lipschitz_constant()
                        tqdm.write("Initial Lipschitz: %.4f " % L)
                    diff = (outputs[:16] - outputs[16:]).pow(2).sum(1).sqrt()
                    input_diff = (inputs[:16] - inputs[16:]).pow(2).sum((1, 2, 3)).sqrt()
                    loss = diff / input_diff.clamp_min(1e-9)
                    (-loss.mean()).backward()
                    optimizer.step()
                    lc.append(loss.max().item())

                    pbar.set_description("Empirical Lipschitz: %.4f" % loss.max().item())

            print("Final Empirical Lipschitz: %.4f" % loss.max().item())


def LoadLine(checkpoint: str):
    json_path = os.path.join(checkpoint, "config.json")
    yaml_path = os.path.join(checkpoint, "config.yaml")
    if os.path.exists(json_path):
        print("Loading config from json file...")
        with open(json_path, "r") as json_file:
            json_dict = json.load(json_file)
        json_dict["checkpoint"] = checkpoint
        json_dict["mode"] = 'multiplot'
        config = argparse.Namespace(**json_dict)
    elif os.path.exists(yaml_path):
        print("Loading config from yaml file...")
        with open(yaml_path + "config.yaml", "r") as yaml_file:
            yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_dict["checkpoint"] = checkpoint
        yaml_dict["mode"] = 'multiplot'
        config = argparse.Namespace(**yaml_dict)
    else:
        raise ValueError(f"No checkpoint folder found:\n {checkpoint}")
    return Evaluator(config)


@dataclass()
class ApproximateAccuracy(object):
    curve: List

    def __post_init__(self):
        self.train_curve = self.curve[0]
        self.test_curve = self.curve[1]
        self.train_correct = self.curve[2]
        self.test_correct = self.curve[3]
        self.acr = (self.test_correct * self.test_curve).mean()

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        return np.array([self.at_radius(radius) for radius in radii])

    def at_radius(self, radius: float):
        return (self.test_correct & (self.test_curve >= radius)).mean()


class Line(object):
    def __init__(self, quantity: ApproximateAccuracy, legend: str, plot_fmt: str = "", scale_x: float = 1, color='r', width=1.5):
        self.quantity = quantity
        self.legend = legend + f'; ACR:{quantity.acr:.3f}'
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x
        self.color = color
        self.width = width


def plot_certified_accuracy(outfile: str, lines: List[Line], title: str, max_radius: float, radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure(figsize=(6, 5))
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, linewidth=line.width, color=line.color)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=16)
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=17)
    plt.tight_layout()
    plt.title(title, fontsize=22)
    plt.tight_layout()

    plt.savefig(outfile + ".pdf", bbox_inches='tight', pad_inches=0.05)
    plt.savefig(outfile + ".png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f'Plot ACR curves to {outfile}')
    plt.close()


def plot_histograms(
    outfile: str,
    lines: List[Line],
):
    plt.figure(figsize=(12, 6))
    for line in lines:
        plt.subplot(1, 2, 1)
        plt.hist(line.quantity.train_curve, bins=list(np.linspace(-2, 2, 60)), alpha=0.9, label='train', histtype="step")
        plt.grid()
        plt.xlabel('Margin', fontsize=15)
        plt.ylabel('number', fontsize=15)
        plt.legend(fontsize=15)
        plt.title('train')

        plt.subplot(1, 2, 2)
        plt.hist(line.quantity.test_curve, bins=list(np.linspace(-2, 2, 60)), alpha=0.9, label='test', histtype="step")
        plt.grid()
        plt.xlabel('Margin', fontsize=15)
        plt.legend(fontsize=15)
        plt.title('test')

    plt.tight_layout()
    plt.savefig(outfile + ".pdf", bbox_inches='tight', pad_inches=0.05)
    plt.savefig(outfile + ".png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f'Plot histogram to {outfile}')
    plt.close()
