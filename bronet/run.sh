#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

# export CUDA_VISIBLE_DEVICES=0
# Require one gpu

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
  --master_port $MASTER_PORT train.py --launcher=pytorch \
  --config="./configs/benchmark_reproduce/cifar10.yaml" \
  --work_dir="./checkpoint/benchmark_reproduce/cifar10/"

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
  --master_port $MASTER_PORT train.py --launcher=pytorch \
  --config="./configs/benchmark_reproduce/cifar100.yaml" \
  --work_dir="./checkpoint/benchmark_reproduce/cifar100/"

# export CUDA_VISIBLE_DEVICES=0,1
# Require two gpus

OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 \
  --master_port $MASTER_PORT train.py --launcher=pytorch \
  --config="./configs/benchmark_reproduce/tiny_imagenet.yaml" \
  --work_dir="./checkpoint/benchmark_reproduce/tiny_imagenet/"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Require eight gpus

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 \
  --master_port $MASTER_PORT train.py --launcher=pytorch \
  --config="./configs/benchmark_reproduce/imagenet.yaml" \
  --work_dir="./checkpoint/benchmark_reproduce/imagenet/"
