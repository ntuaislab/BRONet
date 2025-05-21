import os
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    extract_archive,
    check_integrity,
    download_url,
    verify_str_arg,
)

from . import cifar10_mean, cifar10_std, mu, std


def inverse_image(X):
    return X * std + mu


class ModelNormWrapper(torch.nn.Module):
    def __init__(self, model, means, stds):
        super(ModelNormWrapper, self).__init__()
        self.model = model
        self.means = torch.Tensor(means).float().view(3, 1, 1).cuda()
        self.stds = torch.Tensor(stds).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.means) / self.stds
        return self.model.forward(x)


def get_loaders(dir_, batch_size, dataset_name="cifar10", normalize=True):
    if dataset_name == "cifar10":
        dataset_func = datasets.CIFAR10
        input_size = 32
        padding = 4
        mean, std = cifar10_mean, cifar10_std
    elif dataset_name == "cifar100":
        dataset_func = datasets.CIFAR100
        input_size = 32
        padding = 4
        mean, std = cifar10_mean, cifar10_std
    elif dataset_name == "tinyimg":
        dataset_func = TinyImageNet
        input_size = 64
        padding = 8
        mean, std = cifar10_mean, cifar10_std  # HACK: temporary use cifar statistics for tinyimg
    elif dataset_name.startswith("test"):
        # INFO: for benchmarking purposes, resized CIFAR-10 will be used by passing `test<input_size>` as dataset name, e.g., test32, test64
        try:
            input_size = int(dataset_name[4:])
            assert input_size in [32, 64, 128]
        except ValueError:
            raise ValueError("Dataset name should be in the format 'test<input_size>' where <input_size> is an integer in [32, 64, 128]")
        dataset_func = datasets.CIFAR10
        padding = None  # Doesn't using random crop, so no padding is needed.
        mean, std = cifar10_mean, cifar10_std
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))

    if dataset_name.startswith("test"):
        if normalize:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                ]
            )
    else:
        if normalize:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(input_size, padding=padding),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(input_size, padding=padding),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    num_workers = 4
    if dataset_name != "tinyimg":
        train_dataset = dataset_func(dir_, train=True, transform=train_transform, download=True)
        test_dataset = dataset_func(dir_, train=False, transform=test_transform, download=True)
    else:
        train_dataset = dataset_func(dir_, split="train", transform=train_transform, download=True)
        test_dataset = dataset_func(dir_, split="val", transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


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

    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

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
            raise RuntimeError("Dataset not found. You can use download=True to download it.")
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

    @staticmethod
    def find_classes(class_file):
        with open(class_file) as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    @staticmethod
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


class AuxDDPM(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        # logging.info("Loading auxiliary dataset from: %s..." % file_path)
        print(f"Loading auxiliary dataset from: {file_path}...")
        data = np.load(file_path)
        self.img = data["image"]
        self.label = data["label"]

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_aux_loaders(dir_, batch_size, dataset_name="cifar10", normalize=True, num_workers=4):
    if dataset_name == "cifar10":
        dataset_func = datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_func = datasets.CIFAR100

    if normalize:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    aux_set = AuxDDPM(dir_, transform=train_transform)

    aux_loader = torch.utils.data.DataLoader(
        aux_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
    )

    return aux_loader


def test_get_loaders(dataset_name, expected_size):
    print(f"\nTesting dataset '{dataset_name}' with expected image size {expected_size}x{expected_size}")
    train_loader, test_loader = get_loaders(dir_='./data', batch_size=4, dataset_name=dataset_name, normalize=True)

    train_images, train_labels = next(iter(train_loader))
    print("Train images shape:", train_images.shape)  # Expected: [batch_size, channels, expected_size, expected_size]

    test_images, test_labels = next(iter(test_loader))
    print("Test images shape:", test_images.shape)

    assert train_images.shape[2] == expected_size and train_images.shape[3] == expected_size, "Train image size mismatch"
    assert test_images.shape[2] == expected_size and test_images.shape[3] == expected_size, "Test image size mismatch"
    print("Test passed for", dataset_name)


if __name__ == "__main__":
    test_get_loaders("test32", 32)
    test_get_loaders("cifar10", 32)
    test_get_loaders("test48", 48)
    test_get_loaders("test64", 64)
