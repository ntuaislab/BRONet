#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export CUDA_VISIBLE_DEVICES=0

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
  --master_port $MASTER_PORT test.py --launcher=pytorch \
  --config='./benchmark_reproduce/cifar10/cifar10.yaml' \
  --work_dir='./checkpoint/benchmark_reproduce/cifar10/' \
  --resume_from='./benchmark_reproduce/cifar10/cifar10/cifar10-12x512_799.pth'
