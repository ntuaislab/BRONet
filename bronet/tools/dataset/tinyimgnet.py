import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    extract_archive,
    check_integrity,
    download_url,
    verify_str_arg,
)


class simple_dataset(Data.Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index: int):
        X = Image.fromarray(self.X[index])
        if self.transform is not None:
            X = self.transform(X)
        Y = self.Y[index]
        return X, Y

    def __len__(self):
        return self.X.shape[0]


def tinyimagenet_dataset(data_root="./data"):
    # data = np.load("%s/tiny200.npz" % data_root)
    # trainX = data["trainX"]
    # trainY = data["trainY"]
    # valX = data["valX"]
    # valY = data["valY"]
    #
    # # save memory from uint8 vs float32, do it on the fly
    # # trainX = trainX.float().div_(255.)
    # # valX = valX.float().div_(255.)
    #
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandAugment()]),
            transforms.ToTensor(),
        ]
    )
    #
    # trainset = simple_dataset(trainX, trainY, transform_train)
    # testset = simple_dataset(valX, valY, transforms.ToTensor())
    # return trainset, testset

    trainset = TinyImageNet(
        data_root, split="train", transform=transform_train, download=True
    )
    valset = TinyImageNet(
        data_root, split="val", transform=transforms.ToTensor(), download=True
    )
    return trainset, valset


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again.
    """

    base_folder = "tiny-imagenet-200/"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super(TinyImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(
            split,
            "split",
            (
                "train",
                "val",
            ),
        )

        if self._check_integrity():
            # print('Files already downloaded and verified.')
            pass
        elif download:
            self._download()
        else:
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )
        if not os.path.isdir(self.dataset_path):
            print("Extracting...")
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, "wnids.txt"))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print("Downloading...")
        download_url(self.url, root=self.root, filename=self.filename)
        print("Extracting...")
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == "train":
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, "images")
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, "images")
        imgs_annotations = os.path.join(dir_path, "val_annotations.txt")

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split("\t"), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


def DDPM_dataset(data_root="./data", num_classes=100):
    crop_size, padding = 32, 2

    # data = np.load(f"{data_root}/c{num_classes}_ddpm.npz")
    if num_classes == 10:
        # data = np.load(f"{data_root}/cifar10_edm_1m.npz")
        data = np.load(f"{data_root}/c10_ddpm.npz")
    elif num_classes == 100:
        data = np.load(f"{data_root}/cifar100_edm_1m.npz")
    elif num_classes == 200:
        data = np.load(f"{data_root}/tiny_img_edm_1m.npz")
    else:
        raise ValueError("Invalid number of classes for DDPM dataset!")
    trainX = data["image"]
    trainY = data["label"]
    if num_classes == 200:
        crop_size, padding = 64, 4

    # print(trainX.shape)
    # save memory from uint8 vs float32, do it on the fly
    # trainX = trainX.float().div_(255.)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    trainset = simple_dataset(trainX, trainY, transform_train)
    return trainset
