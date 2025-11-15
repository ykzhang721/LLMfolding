#!/bin/bash

# ==========================================
# Training Launch Script for LLMFolding
# ==========================================
export HYDRA_FULL_ERROR=1

export WANDB_API_KEY=6902a44455da9c60a239576805da294cc5f1414d
# 训练参数
CONFIG_NAME="folding"
MASTER_PORT=29501
NNODES=2
NPROC_PER_NODE=8

# WandB 参数
export WANDB_ENTITY="LLMFolding"
export WANDB_PROJECT="s2s_v1"
export WANDB_RUN_NAME="qwen3-0.6b-s2s-v1-5w-16gpu"

# 打印配置信息
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Config Name:        $CONFIG_NAME"
echo "Master Port:        $MASTER_PORT"
echo "Number of Nodes:    $NNODES"
echo "GPUs per Node:      $NPROC_PER_NODE"
echo "Total GPUs:         $((NNODES * NPROC_PER_NODE))"
echo "=========================================="
echo ""

# 打印 Hydra 配置（完整参数）
echo "=========================================="
echo "Hydra Configuration Preview"
echo "=========================================="
python pipe.py --config-name="$CONFIG_NAME" --cfg job --resolve
echo ""
echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""

# 启动训练
torchrun \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    pipe.py \
    --config-name="$CONFIG_NAME"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with error code: $?"
    echo "=========================================="
    exit 1
fi
