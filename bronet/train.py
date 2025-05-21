import argparse
import os
import time

import timm.optim
import torch
import yaml

import models
import tools
import logging
import glob
from gpustat import GPUStatCollection
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser("Training")

    parser.add_argument("--config", type=str, help="path to the config yaml file")
    # checkpoint saving
    parser.add_argument("--work_dir", default="./checkpoint/", type=str)
    parser.add_argument("--ckpt_prefix", default="", type=str)
    parser.add_argument("--max_save", default=2, type=int)
    parser.add_argument("--resume_from", default="", type=str)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="quick debug mode, only run one batch for each training epoch.",
    )
    # distributed training
    parser.add_argument(
        "--launcher",
        default="slurm",
        type=str,
        help="should be either `slurm` or `pytorch`",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    return parser.parse_args()


def find_latest_checkpoint(folder):
    checkpoint_files = glob.glob(os.path.join(folder, "*.pth"))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint


def main():
    args = get_args()
    os.makedirs(args.work_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # save the cfg file at work_dir
    with open(os.path.join(args.work_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    loss_cfg = cfg["loss"]

    if args.ckpt_prefix == "":
        depth, width = model_cfg["depth"], model_cfg["width"]
        prefix = f"{dataset_cfg['name']}-{depth}x{width}"
        args.ckpt_prefix = prefix

    if args.resume_from:
        if os.path.isdir(args.resume_from):
            args.resume_from = find_latest_checkpoint(args.resume_from)
        print(f"Resume from {args.resume_from}")
        ckpt = torch.load(args.resume_from, "cpu")
        backbone_ckpt = ckpt["backbone"]
        optimizer_ckpt = ckpt["optimizer"]
        start_epoch = ckpt["start_epoch"]
        current_iter = ckpt["current_iter"]
        training_logs = ckpt["training_logs"]
        resume = True
    else:
        start_epoch = 0
        training_logs = []
        resume = False

    rank, local_rank, num_gpus = tools.init_DDP(args.launcher)
    print("Inited distributed training!")

    if local_rank == 0:
        os.system(f"cat {args.config}")

    print(f"Use checkpoint prefix: {args.ckpt_prefix}")

    logfile = os.path.join(args.work_dir, "output.log")
    if not args.resume_from and os.path.exists(logfile):
        os.remove(logfile)

    train_loader, train_sampler, val_loader, _ = tools.data_loader(
        data_name=dataset_cfg["name"],
        batch_size=train_cfg["batch_size"] // num_gpus,
        num_classes=dataset_cfg["num_classes"],
    )
    # print number of samples in train and val
    print(f"Number of samples in train: {len(train_loader) * train_cfg['batch_size']}")
    print(f"Number of samples in val: {len(val_loader) * train_cfg['batch_size']}")

    if dataset_cfg["ddpm"]:
        aug_loader, aug_sampler, _, _ = tools.data_loader(
            data_name="ddpm",
            batch_size=train_cfg["batch_size"] * dataset_cfg["ddpm_ratio"] // num_gpus,
            num_classes=dataset_cfg["num_classes"],
        )
        aug_iter = iter(aug_loader)
        print(f"Number of samples in aug: {len(aug_loader) * train_cfg['batch_size'] * dataset_cfg['ddpm_ratio']}")

    model_type = model_cfg.get("type", "loresnet")
    if model_type == "loresnet":
        model = models.BRONet(**model_cfg, **dataset_cfg)
    else:
        raise NotImplementedError(f"model type {model_type} not implemented")

    if resume:
        model.load_state_dict(backbone_ckpt)
        print("Loaded model from checkpoint")
    print(model)
    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    is_sgd = cfg["training"].get("sgd", False)
    if cfg["training"]["nadam"]:
        optim_fn = torch.optim.NAdam
    elif is_sgd:
        optim_fn = torch.optim.SGD
    else:
        optim_fn = torch.optim.Adam
    if is_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            momentum=train_cfg["momentum"],
        )
    else:
        optimizer = optim_fn(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
    if cfg["training"]["lookahead"]:
        optimizer = timm.optim.Lookahead(optimizer)

    scheduler = tools.lr_scheduler(
        iter_per_epoch=len(train_loader),
        max_epoch=train_cfg["epochs"],
        warmup_epoch=train_cfg["warmup_epochs"],
    )

    if resume:
        optimizer.load_state_dict(optimizer_ckpt)
        scheduler.current_iter = current_iter
        scheduler.base_lr = optimizer_ckpt["param_groups"][0]["initial_lr"]
        sub_lipschitz = model.module.sub_lipschitz()
        if isinstance(sub_lipschitz, torch.Tensor):
            sub_lipschitz = sub_lipschitz.item()

    def eps_fn(epoch):
        ratio = min(epoch / train_cfg["epochs"] * 2, 1)
        ratio = loss_cfg["min_eps_ratio"] + (loss_cfg["max_eps_ratio"] - loss_cfg["min_eps_ratio"]) * ratio
        return loss_cfg["eps"] * ratio

    os.makedirs(args.work_dir, exist_ok=True)

    train_fn = getattr(models, loss_cfg["loss_type"])

    print("Begin Training")
    for log in training_logs:
        print(log)
    if args.resume_from:
        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            filename=os.path.join(args.work_dir, "output.log"),
        )
        with open(logfile, "w") as f:
            for log in training_logs:
                f.write(log + "\n")
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=os.path.join(args.work_dir, "output.log"),
    )
    t = time.time()
    print("log file is saved at", os.path.join(args.work_dir, "output.log"))
    for epoch in range(start_epoch, train_cfg["epochs"]):
        eps = eps_fn(epoch)
        train_sampler.set_epoch(epoch)
        if dataset_cfg["ddpm"]:
            aug_sampler.set_epoch(epoch)
        model.module.set_num_lc_iter(model_cfg["num_lc_iter"])

        model.train()
        correct_vra = correct = total = 0.0
        for idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad(set_to_none=True)
            bs = inputs.shape[0]
            sub_lipschitz = model.module.sub_lipschitz()

            if dataset_cfg["ddpm"]:
                try:
                    input2, target2 = next(aug_iter)
                except StopIteration:
                    aug_sampler.set_epoch(epoch)
                    aug_iter = iter(aug_loader)
                    input2, target2 = next(aug_iter)
                bs_aug = input2.shape[0]

                inputs = torch.cat([inputs, input2])
                targets = torch.cat([targets, target2])

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            y, y_, loss = train_fn(
                model=model,
                x=inputs,
                label=targets,
                lc=sub_lipschitz,
                eps=eps,
                return_loss=True,
                offset=loss_cfg["offset"],
                temperature=loss_cfg["temperature"],
                gamma=loss_cfg["gamma"],
                num_classes=dataset_cfg["num_classes"],
                lip_reg=loss_cfg.get("lip_reg", True),
            )

            _ = scheduler.step(optimizer)
            loss.backward()

            if train_cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_val"])

            optimizer.step()

            correct += y.argmax(1).eq(targets)[:bs].sum().item()
            correct_vra += y_.argmax(1).eq(targets)[:bs].sum().item()
            total += bs
            if args.debug:
                break

        if hasattr(optimizer, "sync_lookahead"):
            optimizer.sync_lookahead()

        if epoch % 5 == 0 or epoch > train_cfg["epochs"] * 0.9:
            if epoch % 100 == 0:
                gpu_stats = GPUStatCollection.new_query()
                for gpu in gpu_stats[:1]:
                    if gpu.processes:
                        string = (
                            f"GPU {gpu.index}: {gpu.name}\n"
                            f" Memory Used: {gpu.memory_used} MB / {gpu.memory_total} MB\n"
                            f" GPU Utilization: {gpu.utilization}%\n"
                            f" Temperature: {gpu.temperature} °C\n"
                            f" Running processes:"
                        )
                        logger.info(string)
                        print(string)
                        for p in gpu.processes:
                            string = (
                                f" PID {p['pid']}: {p['command']} using {p['gpu_memory_usage']} MB\n" f"{'—' * 40}\n"
                            )
                            logger.info(string)
                            print(string)

            model.eval()
            model.module.set_num_lc_iter(500)  # let the power method (for Lip-reg layers) converge
            # only need to comput the sub_lipschitz only once for validation
            sub_lipschitz = 1.0
            if loss_cfg["eps"] != 0:
                sub_lipschitz = model.module.sub_lipschitz()
                if isinstance(sub_lipschitz, torch.Tensor):
                    sub_lipschitz = sub_lipschitz.item()

            val_correct = val_total = 0.0
            val_correct_vra_list = [0.0] * 3

            for inputs, targets in val_loader:
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

            collect_info = [
                correct_vra,
                correct,
                total,
                val_correct_vra_list[0],
                val_correct_vra_list[1],
                val_correct_vra_list[2],
                val_correct,
                val_total,
            ]
            collect_info = torch.tensor(collect_info, dtype=torch.float32, device=inputs.device).clamp_min(1e-9)
            torch.distributed.all_reduce(collect_info)

            acc_train = 100.0 * collect_info[1] / collect_info[2]
            acc_val = 100.0 * collect_info[-2] / collect_info[-1]

            acc_vra_train = 100.0 * collect_info[0] / collect_info[2]
            acc_vra_val_list = [100.0 * collect_info[3 + i] / collect_info[-1] for i in range(3)]
            used = time.time() - t
            if rank == 0:
                string = (
                    f"Epoch {epoch}: "
                    f"Eps:{eps: .2f}; "
                    f"Train:{acc_train: .2f}%,{acc_vra_train: .2f}%; "
                    f"Val:{acc_val: .2f}%,{acc_vra_val_list[0]: .2f}%,{acc_vra_val_list[1]: .2f}%,{acc_vra_val_list[2]: .2f}%; "
                    f"Lip:{sub_lipschitz: .2f}; "
                    f"Time:{used / 60: .2f} mins."
                )
                logger.info(string)
                print(string, end="\r")
                training_logs.append(string)
        else:
            acc_train = acc_val = acc_vra_train = 0.0
            acc_vra_val_list = [0.0] * 3

        if rank == 0:
            state = dict(
                backbone=model.module.state_dict(),
                optimizer=optimizer.state_dict(),
                start_epoch=epoch + 1,
                current_iter=scheduler.current_iter,
                training_logs=training_logs,
                configs=cfg,
            )

            try:
                path = f"{args.work_dir}/{args.ckpt_prefix}_{epoch}.pth"
                torch.save(state, path)
            except PermissionError:
                print("Error saving checkpoint!")
                pass
            if epoch >= args.max_save:
                if epoch - args.max_save not in [
                    train_cfg["epochs"] // 2 - 1,
                    train_cfg["epochs"] // 4 - 1,
                    train_cfg["epochs"] * 3 // 4 - 1,
                    train_cfg["epochs"] - 100 - 1,
                ]:
                    path = f"{args.work_dir}/" f"{args.ckpt_prefix}_{epoch - args.max_save}.pth"
                    os.system("rm -f " + path)

        t = time.time()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    main()
