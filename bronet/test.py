import argparse
import os

import torch
import yaml

import models
import tools

# import logging
import autoattack
from tqdm import tqdm


# logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser("Testing Neural Networks")

    parser.add_argument("--config", type=str, help="path to the config yaml file")
    parser.add_argument("--work_dir", default="./checkpoint/test", type=str)
    parser.add_argument("--resume_from", default="", type=str)
    parser.add_argument(
        "--launcher",
        default="slurm",
        type=str,
        help="should be either `slurm` or `pytorch`",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--norm", default="", type=str)

    return parser.parse_args()


@torch.no_grad()
def test():
    args = get_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    loss_cfg = cfg["loss"]

    rank, local_rank, num_gpus = tools.init_DDP(args.launcher)

    if local_rank == 0:
        os.system(f"cat {args.config}")

    _, _, val_loader, _ = tools.data_loader(
        data_name=dataset_cfg["name"],
        batch_size=train_cfg["batch_size"] // num_gpus,
        num_classes=dataset_cfg["num_classes"],
    )

    model = models.BRONet(**model_cfg, **dataset_cfg)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, "cpu")
        backbone_ckpt = ckpt["backbone"]
        model_dict = model.state_dict()
        for k, v in backbone_ckpt.items():
            if k not in model_dict:
                print(f"Warning: Key {k} from checkpoint not in current model implementation so is not loaded.")
        backbone_ckpt = {k: v for k, v in backbone_ckpt.items() if k in model_dict}
        model.load_state_dict(backbone_ckpt)

    # print(model)
    model = model.cuda()

    print(f"Begin Testing on {dataset_cfg['name']}.")

    model.eval()
    model.set_num_lc_iter(500)

    sub_lipschitz = 1.0
    if loss_cfg["eps"] != 0:
        with torch.no_grad():
            sub_lipschitz = model.sub_lipschitz()
        if isinstance(sub_lipschitz, torch.Tensor):
            sub_lipschitz = sub_lipschitz.item()

    print(f"sub_lipschitz: {sub_lipschitz}")

    val_correct = val_total = 0.0
    val_correct_vra_list = [0.0] * 3

    for inputs, targets in tqdm(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            y, y_list = models.get_predictions(
                model,
                x=inputs,
                eps_list=[loss_cfg["eps"] * i for i in range(1, 4)],
                lc=sub_lipschitz,
            )

            val_correct += y.argmax(1).eq(targets).sum().item()
            for i, y_ in enumerate(y_list):
                val_correct_vra_list[i] += y_.argmax(1).eq(targets).sum().item()
            val_total += targets.size(0)

    acc_val = 100.0 * val_correct / val_total

    acc_vra_val_list = [100.0 * val_correct_vra_list[i] / val_total for i in range(3)]

    string = (
        f"Test results: "
        f"Clean{acc_val: .2f}%, VRA acc{acc_vra_val_list[0]: .2f}%, {acc_vra_val_list[1]: .2f}%, {acc_vra_val_list[2]: .2f}%. "
        f"sub_lipschitz:{sub_lipschitz: .2f}."
    )

    # logger.info(string)
    print(string)
    # use autoattack to evaluate the robustness of the model
    # create logfile
    if args.norm:
        os.makedirs(args.work_dir, exist_ok=True)
        logfile = os.path.join(args.work_dir, f"{args.norm}_autoattack.log")

    if args.norm == "l2":
        eps = 36 / 255.0
        print(f"Evaluate l2 autoattack with eps={eps}")
        adversary = autoattack.AutoAttack(
            model,
            norm="L2",
            eps=eps,
            version="standard",
            log_path=logfile,
        )
        images, labels = [], []
        for image, label in val_loader:
            images.append(image)
            labels.append(label)
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        _ = adversary.run_standard_evaluation(images, labels, bs=train_cfg["batch_size"])

        eps = 72 / 255.0
        print(f"Evaluate l2 autoattack with eps={eps}")
        adversary = autoattack.AutoAttack(
            model,
            norm="L2",
            eps=eps,
            version="standard",
            log_path=logfile,
        )
        images, labels = [], []
        for image, label in val_loader:
            images.append(image)
            labels.append(label)
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        _ = adversary.run_standard_evaluation(images, labels, bs=train_cfg["batch_size"])

        eps = 108 / 255.0
        print(f"Evaluate l2 autoattack with eps={eps}")
        adversary = autoattack.AutoAttack(
            model,
            norm="L2",
            eps=eps,
            version="standard",
            log_path=logfile,
        )
        images, labels = [], []
        for image, label in val_loader:
            images.append(image)
            labels.append(label)
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        _ = adversary.run_standard_evaluation(images, labels, bs=train_cfg["batch_size"])
    elif args.norm == "linf":
        eps = 8 / 255.0
        print(f"Evaluate linf autoattack with eps={eps}")
        adversary = autoattack.AutoAttack(
            model,
            norm="Linf",
            eps=eps,
            version="standard",
            log_path=logfile,
        )
        images, labels = [], []
        for image, label in val_loader:
            images.append(image)
            labels.append(label)
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        _ = adversary.run_standard_evaluation(images, labels, bs=train_cfg["batch_size"])


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    test()
