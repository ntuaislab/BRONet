# LipConvnet

## Run

### Prerequisites

The root folder for datasets is `./data` by default.

### Training Command

To reproduce the results for LipConvNet-10-32 with the BRO layer, run the command below.
For more details on available arguments, refer to `main.py`.

```bash
python main.py \
    --dataset cifar100 \
    --conv-layer bro \
    --block-size 2 \
    --init-channels 32 \
    --out-dir ./exp/LipConvnet \
```

### Note

We also support other orthogonal convolution methods, such as Cayley, SOC and LOT.
