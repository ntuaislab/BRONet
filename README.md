# Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss



## 🚂 Overview

Official PyTorch implementation for our ICML 2025 spotlight paper. We introduce:

- **Block Reflector Orthogonal Layer (BRO):** A low-rank, approximation-free orthogonal convolutional layer designed to efficiently construct Lipschitz neural networks, improving both stability and expressiveness.
- **Logit Annealing Loss (LA):** An adaptive loss function that dynamically balances classification margins across samples, leading to enhanced certified robustness.

## 📁 Repository Structure

- `bronet/` — Contains the implementation for BRONet experiments.
- `lipconvnet/` — Contains the implementation for the LipConvNet experiments.

Key modules:

- **BRO layer:** [`lipconvnet/models/layers/bro.py`](./lipconvnet/models/layers/bro.py)
- **LA loss:** [`bronet/models/margin_layer.py`](./bronet/models/margin_layer.py)

## 🚀 Getting Started

To set up the environment and run our code:

### 1. Requirements

- Python 3.11
- PyTorch ≥ 2.0 with CUDA support
- A recent NVIDIA GPU (e.g., Ampere or newer) is recommended for training and certification

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ntuaislab/BRONet.git
cd BRONet
pip install -r requirements.txt
```

### 3. Reproduce the paper results

To reproduce the main results in the paper, run the following command:

```bash
cd bronet
bash run.sh
```

## 🎯 Pre-trained Models

|           Datasets            |    Models    |                                Checkpoint                                |
| :---------------------------: | :----------: | :----------------------------------------------------------------------: |
|      ImageNet (Table 1)       | BRONet (+LA) |   [Link](https://huggingface.co/pinhank121/BRONet_ImageNet/tree/main)    |
| ImageNet w/ EDM2 2M (Table 2) | BRONet (+LA) | [Link](https://huggingface.co/pinhank121/BRONet_ImageNet_EDM2/tree/main) |

To test the provided models, download the checkpoint and config file, then run:

```bash
cd bronet
python test.py --resume_from="path_to_downloaded_checkpoint" --config="path_to_config"
```

See [`bronet/README.md`](./bronet/README.md) for instructions on reproducing the results.

## 🤝 Acknowledgements

This work builds on and benefits from several open-source efforts:

- [Cayley Layer](https://github.com/locuslab/orthogonal-convolutions)
- [SOC](https://github.com/singlasahil14/SOC)
- [LOT](https://github.com/AI-secure/Layerwise-Orthogonal-Training)
- [SLL](https://github.com/araujoalexandre/Lipschitz-SLL-Networks)
- [LiResNet](https://github.com/hukkai/liresnet)

We sincerely thank the authors of these projects for making their work publicly available.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 📄 Citation

If you find our work useful, please cite us:

```bibtex
@inproceedings{lai2025enhancing,
    title={Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss},
    author={Bo-Han Lai and Pin-Han Huang and Bo-Han Kung and Shang-Tse Chen},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025},
    note={Spotlight}
}
```
