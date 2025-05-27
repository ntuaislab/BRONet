# Block Reflector Orthogonal Neural Network (BRONet)

Code in this repository is adapted from [code repo of LiResNet](https://github.com/hukkai/liresnet/tree/main)

## Prerequisites

- The root folder for datasets is `./data` by default.
- For experiments with diffusion augmentation experiments (Table 2):
  - Download CIFAR-10 EDM 4M released by: [LiResNet](https://github.com/hukkai/liresnet/tree/main), and move it to `./data/c10_ddpm.npz`.
  - CIFAR-100 EDM 1M released by: [DM-Improves-AT](https://github.com/wzekai99/DM-Improves-AT), and move it to `./data/cifar100_edm_1m.npz`.
  - ImageNet EDM2 (+AutoGuidance) 2M: Please clone the repo [edm2](https://github.com/NVlabs/edm2) and use the checkpoint `edm2-img512-xxl-autog-dino` to generate 2 Million 512x512 images. The generation script should be adapted so that the images are saved under subfolders named after class idx (eg., `./out/0/`, `./out/1/`, ...). The adapted generation scripts and commands are provided in `edm2_gen_script/`, which can be used in your cloned `edm2/` repo. After generation, the `bronet` training scripts expect the images and class subfolders are placed in `./data/imagenet_ddpm/`.

## Run

To train a BRONet model, use the `train.py` script with a configuration file.
See `run.sh` for example usage of the training commands.

Remember to adjust `nproc_per_node` to the number of GPUs accordingly.
To reproduce the results in Table 1:

```bash
bash run.sh
```

### Available Configurations

The repository includes several pre-defined configurations in the `configs/benchmark_reproduce/` directory:

- `cifar10.yaml`: Configuration for CIFAR-10 dataset BRONet-L (+LA) (1 GPU required)
- `cifar100.yaml`: Configuration for CIFAR-100 dataset BRONet-L (+LA) (1 GPU required)
- `cifar10_edm.yaml`: Configuration for CIFAR-10 with EDM data augmentation (Table 2) (2 GPUs required)
- `cifar100_edm.yaml`: Configuration for CIFAR-100 with EDM data augmentation (Table 2) (2 GPUs required)
- `tiny_imagenet.yaml`: Configuration for Tiny ImageNet dataset BRONet (+LA) (2 GPUs required)
- `imagenet.yaml`: Configuration for ImageNet dataset BRONet (+LA) (8 GPUs required)
- `imagenet_edm2.yaml`: Configuration for ImageNet with EDM2 data augmentation (Table 2) (8 GPUs required)
