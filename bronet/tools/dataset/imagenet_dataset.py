from typing import Tuple

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os as os
from PIL import Image


class UniformResize(torch.nn.Module):
    def __init__(self, lower: int, upper: int):
        super(UniformResize, self).__init__()
        self.lower = lower
        self.upper = upper + 1

    def forward(self, x):
        size = torch.randint(self.lower, self.upper, size=[]).item()
        return transforms.Resize(size)(x)


def _imgnet_dataset(data_root: str = "./data/", mode: str = "train"):
    if mode == "train":
        transform = transforms.Compose(
            [
                UniformResize(224, 288),
                transforms.RandAugment(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageNet(
            os.path.join(data_root, "imagenet2012"),
            split="train",
            transform=transform,
        )

    elif mode == "val":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        dataset = datasets.ImageNet(
            os.path.join(data_root, "imagenet2012"),
            split="val",
            transform=transform,
        )
    elif mode == "ddpm":
        transform = transforms.Compose(
            [
                UniformResize(224, 288),
                transforms.RandAugment(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = DDPMFolderDataset(
            os.path.join(data_root, "imagenet_ddpm/"),
            transform=transform,
        )
    else:
        raise ValueError("Invalid mode")

    return dataset


def imagenet_dataset(data_root: str, ddpm: bool = False):
    ddpm_dataset = train_dataset = val_dataset = None
    if ddpm:
        ddpm_dataset = _imgnet_dataset(data_root, mode="ddpm")
    else:
        train_dataset = _imgnet_dataset(data_root, mode="train")
        val_dataset = _imgnet_dataset(data_root, mode="val")

    return train_dataset, val_dataset, ddpm_dataset


class DDPMFolderDataset(Data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        The subfolder name is expected to be the class label.
        Args:
            root_dir (str): Path to the dataset root directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Iterate over subdirectories
        for class_name in sorted(os.listdir(root_dir)):  # Ensure consistent ordering
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):  # Check if it's a directory
                label = int(class_name)  # Convert folder name to int label
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith((".png", ".jpg", ".jpeg")):  # Check if it's an image
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
        if self.transform:
            image = self.transform(image)
        return image, label

