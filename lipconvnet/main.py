import warnings
import argparse
import yaml
import json
import os

from trainer import Trainer, Evaluator

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()

    # Training specifications
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr-min", default=0.0, type=float)
    parser.add_argument("--lr-max", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=3e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--gamma", default=0.0, type=float, help="gamma for certificate regularization")
    parser.add_argument("--amp_type", default="float32", type=str, choices=["bfloat16", "float16", "float32"], help="choose amp data type: bfloat16 or float16")
    parser.add_argument("--loss-scale", default="1.0", type=str, choices=["1.0", "dynamic"], help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument("--lr-scheduler", default="default", type=str, choices=["default", "cosine", "cosine-wr", "triangle", "custom_cosine"], help="learning rate schedule")
    parser.add_argument("--custom_warmup_epoch", default=10, type=int, help="number of warmup epochs for custom cosine learning rate scheduler.")
    parser.add_argument("--act-lr-factor", default=1.0, type=float, help="learning rate factor for activation")
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adam", "adamw", "radam", "nadam"], help="optimizer")
    parser.add_argument("--cr-scheduler", default="default", type=str, choices=["default", "linear", "quad", "sqrt", "cosine", "cosine-alt"], help="CR scheduler")
    parser.add_argument("--loss", default="la", type=str, choices=["ce", "xent", "la"], help="loss function")
    parser.add_argument("--lookahead", action="store_true", help="use lookahead optimizer")

    # LA loss specifications (default values are tuned on CIFAR-100)
    parser.add_argument("--temperature", default=0.75, type=float, help="temperature for softmax in xent loss")
    parser.add_argument("--offset", default=2.0, type=float, help="factor of offset for softmax in xent loss")
    parser.add_argument("--la_beta", default=5.0, type=float, help="gamma for focal loss")
    parser.add_argument("--la_alpha", default=0.25, type=float, help="alpha for focal loss")

    # Genral Model architecture specifications
    parser.add_argument("--conv-layer", default="bro", type=str, choices=["cayley", "soc", "standard", "lot", "bro"])
    parser.add_argument("--bro-rank", default=0.125, type=float)
    parser.add_argument("--dense-layer", default="bro", type=str, choices=["cayley", "bro"])
    parser.add_argument("--init-channels", default=32, type=int)
    parser.add_argument("--activation", default="maxmin", help="Activation function", choices=['maxmin', 'hh1', 'hh2', 'relu', 'rrelu', 'leakyrelu', 'prelu', 'softplus', 'vrelu', 'plrelu'])
    parser.add_argument("--block-size", default=2, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="size of each block")
    parser.add_argument("--lln", action="store_true", help="set last linear to be linear and normalized")
    parser.add_argument("--kernel", default=3, type=int)
    parser.add_argument("--num-dense", default=0, type=int)
    parser.add_argument("--num-conv", default=1, type=int)

    # Dataset specifications
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--dataset", default="cifar100", type=str, help="dataset to use for training")
    parser.add_argument("--original_ratio", default=1, type=int, help="The multiplier for original data batch size")
    parser.add_argument("--edm_ratio", default=0, type=int, help="The multiplier for ddpm data batch size")

    # Other specifications
    parser.add_argument("--out-dir", default="./exp/LipConvnet", type=str, help="Output directory")
    parser.add_argument("--seed", default=3407, type=int, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="debug mode, wont save anything")

    # Evaluation specifications
    parser.add_argument("--checkpoint", type=str, help="load test model directory")
    parser.add_argument("--epsilon", default=36, type=int)

    # Mode specifications
    parser.add_argument("--mode", default="train", type=str, choices=["train", "certified", "auto-atk", "pgd-atk", "upper-lipschitz", "lower-lipschitz"])
    parser.add_argument("--model", default="lipconvnet", type=str, choices=["lipconvnet"])
    parser.add_argument("--comment", default=None, type=str, help="comment for the experiment")
    return parser.parse_args()


def main(config):
    if config.mode == "train":
        trainer = Trainer(config)
        trainer()
    elif config.mode in ["certified", "pgd-atk", "auto-atk", "upper-lipschitz", "lower-lipschitz"]:
        evaluate = Evaluator(config)
        evaluate()


if __name__ == "__main__":
    config = get_args()
    if config.checkpoint is not None:
        json_path = os.path.join(config.checkpoint, "config.json")
        yaml_path = os.path.join(config.checkpoint, "config.yaml")
        if os.path.exists(json_path):
            print("Loading config from json file...")
            with open(json_path, "r") as json_file:
                json_dict = json.load(json_file)
            config = vars(config)
            print(config.get("mode"))
            json_dict["mode"] = config.get("mode")
            json_dict["checkpoint"] = config.get("checkpoint")
            json_dict["epsilon"] = config.get("epsilon")
            config = argparse.Namespace(**json_dict)
        elif os.path.exists(yaml_path):
            print("Loading config from yaml file...")
            with open(yaml_path + "config.yaml", "r") as yaml_file:
                yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
            config = vars(config)
            yaml_dict["mode"] = config.get("mode")
            yaml_dict["checkpoint"] = config.get("checkpoint")
            yaml_dict["epsilon"] = config.get("epsilon")
            config = argparse.Namespace(**yaml_dict)
        else:
            print("No config file found in checkpoint directory")

    main(config)
