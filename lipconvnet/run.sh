#!/bin/bash
CUDA_VISIBLE_DEVICES=0

for conv in bro soc lot; do
    python main.py \
        --dataset cifar100 \
        --conv-layer ${conv} \
        --block-size 2 \
        --init-channels 32 \
        --out-dir ./exp/LipConvnet \
        --loss la
done
