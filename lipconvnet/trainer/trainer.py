import os
import csv
import time
import json
import yaml
import torch
import logging
import gpustat
import numpy as np
import timm.optim
from itertools import cycle

from utils import cifar10_std
from utils.misc import increment_path, get_git_hash, get_git_commit_msg, get_git_timestamp, get_parameter_lists, loss_mapping, cr_scheduler_mapping, lr_scheduler_mapping, Timer
from utils.data import get_loaders, get_aux_loaders
from utils.eval import (
    ortho_certificates,
    lln_certificates,
    evaluate_certificates,
    robust_statistics,
)


from .base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.logger = self.configure_logger()
        self.train_loader, self.test_loader, self.aux_loader, self.std = self.configure_data_loader()

    def configure_logger(self):
        logger = logging.getLogger(__name__)
        if self.config.debug:
            self.config.out_dir += "_debug"
            print("=== Debug mode, will not save anything ===")
            os.makedirs(self.config.out_dir, exist_ok=True)
        else:
            self.config.out_dir += f"_{self.config.dataset}_{self.config.block_size}_{self.config.conv_layer}_{self.config.activation}_cr{self.config.gamma}"
            self.config.out_dir += f"{'_lln' if self.config.lln else ''}"

            self.config.out_dir = increment_path(self.config.out_dir, exist_ok=False, sep="#")
            os.makedirs(self.config.out_dir, exist_ok=True)
            print(f"Files are saved: {self.config.out_dir}")

        self.config_dict = vars(self.config)

        with open(os.path.join(self.config.out_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config_dict, f, indent=4)
        with open(os.path.join(self.config.out_dir, "config.json"), "w") as f:
            json.dump(self.config_dict, f, indent=4)

        logfile = os.path.join(self.config.out_dir, "output.log")
        if os.path.exists(logfile):
            os.remove(logfile)

        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            filename=os.path.join(self.config.out_dir, "output.log"),
        )
        logger.info(self.config)

        logger.info(f'Experiment Comment: "{self.config.comment}"')
        logger.info(f'Git Hash/Message/Timestamp: {get_git_hash()}/{get_git_timestamp()}/"{get_git_commit_msg()}"')
        return logger

    def configure_data_loader(self):
        if self.config.original_ratio:
            train_loader, test_loader = get_loaders(self.config.data_dir, self.config.batch_size * self.config.original_ratio, self.config.dataset)
        else:
            train_loader, test_loader = [], []

        std = cifar10_std
        if self.config.dataset == "cifar10":
            self.config.num_classes = 10
        elif self.config.dataset == "cifar100":
            self.config.num_classes = 100
            if not self.config.lln:
                print("LLN is automatically enabled for CIFAR-100😎")
            self.config.lln = True
        elif self.config.dataset == "tinyimg":
            self.config.num_classes = 200
            if not self.config.lln:
                print("LLN is automatically enabled for TinyImagnet😎")
            self.config.lln = True
        else:
            # INFO: for benchmarking purposes, resized CIFAR-10 will be used by passing `test<input_size>` as dataset name, e.g., test32, test64
            self.config.num_classes = 10

        aux_loader = None
        if self.config.edm_ratio:
            torch.multiprocessing.set_sharing_strategy("file_system")
            path_to_edm = os.path.join(self.config.data_dir, "cifar100_1m_edm.npz") if self.config.dataset == "cifar100" else os.path.join(self.config.data_dir, "cifar10_1m_edm.npz")
            aux_loader = get_aux_loaders(
                path_to_edm,
                self.config.batch_size * self.config.edm_ratio,
                dataset_name=self.config.dataset,
                normalize=True,
                num_workers=4,
            )
            aux_loader = cycle(aux_loader)
        return train_loader, test_loader, aux_loader, std

    def configure_optimizers(self, model):
        conv_params, activation_params, other_params = get_parameter_lists(model)
        if self.config.optimizer == "sgd":
            opt = torch.optim.SGD(
                model.parameters(),
                lr=self.config.lr_max,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        if self.config.lookahead:
            opt = timm.optim.Lookahead(opt)
        return opt

    @staticmethod
    def model_summary(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = round(sum([np.prod(p.size()) for p in model_parameters]) / 1_000_000)
        # print(f"Model has {params}M trainable parameters")
        return params

    def __call__(self):
        if self.config.num_dense > 0 and self.config.amp_type != "float32":
            raise Exception("lu_factor_cusolver in torch.linalg.solve() not implemented for BFloat16 and float16.")

        if self.config.num_dense > 0 and not self.config.lln:
            print("LLN is automatically enabled when using dense layer😎")
            self.config.lln = True

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        # FIXME:
        # torch.backends.cuda.preferred_linalg_library("cusolver")
        # TODO: set flags for reproducibility
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.determinisc = True

        # Evaluation at early stopping
        model = self.init_model().cuda()
        model.train()

        opt = self.configure_optimizers(model)

        criterion = loss_mapping(self.config.loss, self.config)

        lr_steps = self.config.epochs * len(self.train_loader)

        scheduler = lr_scheduler_mapping(self.config.lr_scheduler, opt, lr_steps, self.config.lr_max, self.config.epochs, self.config.custom_warmup_epoch)

        best_model_path = os.path.join(self.config.out_dir, "best.pth")
        last_model_path = os.path.join(self.config.out_dir, "last.pth")
        last_opt_path = os.path.join(self.config.out_dir, "last_opt.pth")

        # Training
        std = torch.tensor(self.std).cuda()
        L = 1 / torch.max(std)
        hybrid = clean = robust = 0
        rec_hybrid = rec_clean = rec_robust = []
        start_train_time = time.time()
        self.logger.info(
            "Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t " + "Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Test Cert \t Lipschitz",
        )
        timer = Timer(total_epochs=self.config.epochs, moving_average_window=5)

        for epoch in range(self.config.epochs):
            model.train()
            start_epoch_time = time.time()
            train_loss = 0
            train_cert = 0
            train_robust = 0
            train_acc = 0
            train_n = 0

            for _, (X, y) in enumerate(self.train_loader):
                if self.config.edm_ratio:
                    img_aux, label_aux = next(self.aux_loader)
                    X = torch.vstack([X, img_aux])
                    y = torch.hstack([y, label_aux])

                X, y = X.cuda(), y.cuda()

                output = model(X)
                curr_correct = output.max(1)[1] == y
                # model_lipschitz = model.set_and_get_lipschitz_constant()
                # L = 1 / torch.max(std) * model_lipschitz
                if self.config.lln:
                    curr_cert = lln_certificates(output, y, model.last_layer, L)
                else:
                    curr_cert = ortho_certificates(output, y, L)

                loss = criterion(output, y)

                if self.config.gamma > 0:
                    loss = cr_scheduler_mapping(
                        self.config.cr_scheduler,
                        loss,
                        epoch,
                        self.config.epochs,
                        curr_cert,
                        self.config.gamma,
                    )

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * y.size(0)
                train_cert += (curr_cert * curr_correct).sum().item()
                train_robust += ((curr_cert > (self.config.epsilon / 255.0)) * curr_correct).sum().item()
                train_acc += curr_correct.sum().item()
                train_n += y.size(0)
                if self.config.lr_scheduler == "triangle":
                    step = epoch * len(self.train_loader)
                    scheduler.step(step)
                elif self.config.lr_scheduler != "plateau":
                    scheduler.step()

            # Check current test accuracy of model
            # model_lipschitz = model.set_and_get_lipschitz_constant()
            # L = 1 / torch.max(std) * model_lipschitz
            losses_arr, correct_arr, certificates_arr = evaluate_certificates(
                self.test_loader,
                model,
                L,
            )

            test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
                losses_arr,
                correct_arr,
                certificates_arr,
            )

            if self.config.lr_scheduler == "plateau":
                scheduler.step(test_loss)

            _hybrid = test_acc * test_robust_acc_list[0]
            _clean = test_acc
            _robust = test_robust_acc_list[0]

            if _hybrid >= hybrid:
                torch.save(model.state_dict(), best_model_path)
                hybrid = _hybrid
                best_epoch_hybrid = epoch
                rec_hybrid = [
                    test_acc,
                    test_robust_acc_list[0],
                    test_robust_acc_list[1],
                    test_robust_acc_list[2],
                    test_cert,
                ]
            if _clean >= clean:
                clean = _clean
                best_epoch_clean = epoch
                rec_clean = [
                    test_acc,
                    test_robust_acc_list[0],
                    test_robust_acc_list[1],
                    test_robust_acc_list[2],
                    test_cert,
                ]
            if _robust >= robust:
                robust = _robust
                best_epoch_robust = epoch
                rec_robust = [
                    test_acc,
                    test_robust_acc_list[0],
                    test_robust_acc_list[1],
                    test_robust_acc_list[2],
                    test_cert,
                ]

            epoch_time = time.time()

            if self.config.lr_scheduler == "plateau":
                lr = scheduler._last_lr.pop()
            elif self.config.lr_scheduler == "triangle":
                lr = scheduler.get_last_lr()
            elif self.config.lr_scheduler == "custom_cosine":
                lr = scheduler.get_last_lr()
            else:
                lr = scheduler.get_last_lr()[0]
            self.logger.info(
                "%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
                epoch,
                epoch_time - start_epoch_time,
                lr,
                train_loss / train_n,
                train_acc / train_n,
                test_loss,
                test_acc,
                test_robust_acc_list[0],
                test_robust_acc_list[1],
                test_robust_acc_list[2],
                test_cert,
            )
            remaining, past = timer.remaining_time(epoch)
            print(
                f"""🚂{epoch+1:>3}/{self.config.epochs} | {past}<-{remaining} | 0/36/72/108/tACR: {test_acc}/{test_robust_acc_list[0]}/{test_robust_acc_list[1]}/{test_robust_acc_list[2]}/{test_cert:.5f} ({rec_hybrid[0]},{rec_hybrid[1]}) ({rec_robust[0]},{rec_robust[1]})""",
                end="    \r",
            )

            if epoch == self.config.epochs // 2:
                gpus = gpustat.new_query()
                if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                    gpu = gpus[int(os.environ["CUDA_VISIBLE_DEVICES"])]
                else:
                    gpu = gpus[0]
                memory_used = gpu.memory_used

            if epoch == self.config.epochs - 1:
                torch.save(model.state_dict(), last_model_path)
                trainer_state_dict = {
                    "epoch": epoch,
                    "optimizer_state_dict": opt.state_dict(),
                }
                torch.save(trainer_state_dict, last_opt_path)

        train_time = time.time()

        self.logger.info("Total train time: %.4f minutes", (train_time - start_train_time) / 60)
        self.logger.info(f"Memory used: {memory_used} MB")
        self.logger.info(f"Model size: {self.model_summary(model)}M")

        # Evaluation at best model (early stopping)
        model_test = self.init_model().cuda()
        model_test.load_state_dict(torch.load(last_model_path))
        model_test.float()
        model_test.eval()

        start_test_time = time.time()
        losses_arr, correct_arr, certificates_arr = evaluate_certificates(self.test_loader, model_test, L)
        total_time = time.time() - start_test_time

        test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(losses_arr, correct_arr, certificates_arr)

        self.logger.info(
            "Last Epoch \t Test Loss \t Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Mean Cert \t Test Time",
        )
        self.logger.info(
            "%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
            epoch,
            test_loss,
            test_acc,
            test_robust_acc_list[0],
            test_robust_acc_list[1],
            test_robust_acc_list[2],
            test_cert,
            total_time,
        )

        # Evaluation at best model (early stopping)
        model_test.load_state_dict(torch.load(best_model_path))
        model_test.float()
        model_test.eval()

        start_test_time = time.time()
        losses_arr, correct_arr, certificates_arr = evaluate_certificates(self.test_loader, model_test, L)
        total_time = time.time() - start_test_time

        test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(losses_arr, correct_arr, certificates_arr)

        self.logger.info("Best Epoch \t Test Loss \t Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Mean Cert \t Test Time")
        self.logger.info(
            "%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
            best_epoch_hybrid,
            test_loss,
            test_acc,
            test_robust_acc_list[0],
            test_robust_acc_list[1],
            test_robust_acc_list[2],
            test_cert,
            total_time,
        )
        self.logger.info("Record of best epoch")
        self.logger.info("Test Acc/Test Robust (36)/Test Robust (72)/Test Robust (108)/Mean Cert")
        self.logger.info(
            "%.4f, %.4f, %.4f, %.4f, %.4f",
            test_acc,
            test_robust_acc_list[0],
            test_robust_acc_list[1],
            test_robust_acc_list[2],
            test_cert,
        )

        recfile = os.path.join(self.config.out_dir, "output.csv")
        with open(recfile, "w") as f:
            write = csv.writer(f)
            write.writerow(rec_hybrid)
            write.writerow(rec_clean)
            write.writerow(rec_robust)
        print()


if __name__ == "__main__":
    pass
