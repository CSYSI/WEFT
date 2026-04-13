#!/usr/bin/env bash

CONFIG=$1  # 训练的配置文件
GPUS=$2  # GPU 数量
PORT=${PORT:-29500}  # 端口，默认 29500，避免端口冲突可更改

# ✅ 解决 NCCL 可能的 deadlock
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

# ✅ 使用 `torchrun` 启动 `DDP`
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --deterministic ${@:3}