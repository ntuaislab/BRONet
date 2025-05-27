#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

# Generate 50000 images using 8 GPUs and save them as out/*/*.png
# Please use it in your cloned edm2 repository
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --master_port $MASTER_PORT generate_images_subfolder_save.py \
  --preset=edm2-img512-xxl-autog-dino --outdir=out --subdirs --seeds=0-2000000
