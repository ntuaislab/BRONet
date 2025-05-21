# Block Reflector Orthogonal Neural Network (BRONet)

Code in this repository is adapted from [code repo of LiResNet](https://github.com/hukkai/liresnet/tree/main)

## Run

### Prerequisites
- The root folder for datasets is `./data` by default.
- For experiments with diffusion augmentation experiments (Table 2):
    - Download CIFAR-10 EDM 4M released by: [LiResNet](https://github.com/hukkai/liresnet/tree/main), and move it to `./data/c10_ddpm.npz`. 
    - CIFAR-100 EDM 1M released by: [DM-Improves-AT](https://github.com/wzekai99/DM-Improves-AT), and move it to `./data/cifar100_edm_1m.npz`.
    - ImageNet EDM2 (+AutoGuidance) 2M: Please use the checkpoint `edm2-img512-xxl-autog-dino` and generation script in [EDM2](https://github.com/NVlabs/edm2) to generate 2 Million 512x512 images. The images are expected to be placed under `./data/imagenet_ddpm/` with subfolders named after class idx (eg., `./data/imagnet_ddpm/0/`, `./data/imagnet_ddpm/1/`, ...).

### Training Command
See `run.sh` for training commands.

To train a BRONet model, use the `train.py` script with a configuration file:

```bash
python train.py --config configs/benchmark_reproduce/cifar10.yaml --work_dir ./checkpoint/bronet_cifar10
```

### Available Configurations
The repository includes several pre-defined configurations in the `configs/benchmark_reproduce/` directory:
- `cifar10.yaml`: Configuration for CIFAR-10 dataset BRONet-L (+LA)
- `cifar100.yaml`: Configuration for CIFAR-100 dataset BRONet-L
- `cifar10_edm.yaml`: Configuration for CIFAR-10 with EDM data augmentation (Table 2)
- `cifar100_edm.yaml`: Configuration for CIFAR-100 with EDM data augmentation (Table 2)
- `tiny_imagenet.yaml`: Configuration for Tiny ImageNet dataset BRONet (+LA)
- `imagenet.yaml`: Configuration for ImageNet dataset BRONet (+LA)
- `imagenet_edm2.yaml`: Configuration for ImageNet with EDM2 data augmentation (Table 2)
