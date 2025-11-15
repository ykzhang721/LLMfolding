#!/bin/bash

# ==========================================
# Training Launch Script for LLMFolding
# ==========================================

set -e  # 有报错就退出（方便排错）

# -------- 调试 / Hydra --------
export HYDRA_FULL_ERROR=1

# -------- NCCL / CUDA / 线程设置（从 Slurm 脚本里抽的）--------
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}      # H800 那套
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-16}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-1800}

# 线程数（按需改）
CPU_THREADS=${CPU_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$CPU_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$CPU_THREADS}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$CPU_THREADS}

# -------- WandB --------
export WANDB_API_KEY=6902a44455da9c60a239576805da294cc5f1414d
export WANDB_IGNORE_GIT=True
export WANDB_INSECURE_DISABLE_SSL=True

# 如果你有 entity / project 就在这里配
export WANDB_ENTITY="${WANDB_ENTITY:-LLMFolding}"
export WANDB_PROJECT="${WANDB_PROJECT:-s2s_v1}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-0.6b-s2s-v1-5w-8gpu}"

# -------- 训练参数 --------
CONFIG_NAME="folding"
MASTER_PORT=${MASTER_PORT:-29501}

# GPU / 节点数设置（支持外部覆盖）
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-2}

# 为了兼容你下面 echo 的写法，这里单独定义 NPROC_PER_NODE
# 不改 echo 文案，但让它有值可用
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

# 单机情况下 master 地址直接用 localhost
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# -------- 打印配置信息（保持你原来的 echo 不变）--------
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

# -------- 启动训练（本地 torchrun 多卡）--------
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pipe.py \
    --config-name="$CONFIG_NAME"

# 检查训练是否成功
ret=$?
if [ $ret -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with error code: $ret"
    echo "=========================================="
    exit $ret
fi