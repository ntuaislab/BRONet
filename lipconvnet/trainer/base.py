from models.lip_convnets import LipConvNet


class BaseTrainer:
    def __init__(self, config):
        self.config = config

    def init_model(self):
        input_size = 32 if self.config.dataset in ["cifar10", "cifar100"] else 64

        if "test" in self.config.dataset:
            # INFO: for benchmarking purposes, resized CIFAR-10 will be used by passing `test<input_size>` as dataset name, e.g., test32, test64
            try:
                input_size = int(self.config.dataset_name[4:])
                assert input_size in [32, 64, 128]
            except ValueError:
                raise ValueError("Dataset name should be in the format 'test<input_size>' where <input_size> is an integer in [32, 64, 128]")

        if self.config.model == 'lipconvnet':
            model = LipConvNet(
                self.config.conv_layer,
                self.config.activation,
                init_channels=self.config.init_channels,
                block_size=self.config.block_size,
                num_classes=self.config.num_classes,
                lln=self.config.lln,
                kernel_size=self.config.kernel,
                num_dense=self.config.num_dense,
                mask_level=self.config.bro_rank,
                input_size=input_size,
            )
        else:
            raise ValueError(f'Unknown model: {self.config.model}')

        return model


if __name__ == "__main__":
    pass
