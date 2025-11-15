#!/bin/bash

# Specify srun params to start enroot:
#SBATCH --account=protein
#SBATCH --partition=AISS2024110101      # GenSI-protein-lf
#SBATCH --job-name=lf-ar                # job name
#SBATCH --output=output.log             # stdout
#SBATCH --error=error.log               # stderr
#SBATCH --nodes=4                       # nodes
#SBATCH --gres=gpu:8                    # GPUs/node
#SBATCH --ntasks-per-node=1             # tasks/node
#SBATCH --cpus-per-task=224             # CPUs/task
#SBATCH --mem=2000                      # memory(MB)/node
#SBATCH --export=ALL
#SBATCH -w cp2-dgx-[007,031,011,012]

CONTAINER_PATH=/home/projects/protein/lutianyu/images/modern.sqsh
CONTAINER_NAME=modern

# Optimized NCCL and CUDA settings for H800 GPUs:
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3             # Optimized for H800
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=16   # Increased for H800
export NCCL_ALGO=Ring                   # Explicitly set to Ring algorithm
export NCCL_P2P_DISABLE=0               # Enable NCCL peer-to-peer communication (better for H800)
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export VECLIB_MAXIMUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
export TORCH_DISTRIBUTED_TIMEOUT=1800   # Increase distributed training timeout (seconds)


# Check environment:
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Optimized torchrun training script:
TRAIN_SCRIPT=$(cat <<'EOF'

echo "=== Node Information ==="
hostname

echo "=== Environment variables ==="
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "TORCH_DISTRIBUTED_TIMEOUT: $TORCH_DISTRIBUTED_TIMEOUT"

echo "=== GPU visibility ==="
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)

export HTTPS_PROXY="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export HTTP_PROXY="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export https_proxy="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export http_proxy="http://cityu:upside_tumbling_turbine@cp2-fwproxy-vip.aisc.local:3128"
export WANDB_API_KEY=bc2e2b14aacbadfd88a86ceab37243b8944b0eaf
export WANDB_IGNORE_GIT=True
export WANDB_INSECURE_DISABLE_SSL=True

echo "=== Changing pipe host ==="
pip config list
pip config set global.index-url https://pypi.org/simple
pip config list

PIP=/root/miniconda3/envs/qwen3/bin/pip
export TMPDIR=/GenSIvePFS/users/lutianyu/tmp
mkdir -p $TMPDIR

for pkg in colorlog ray einx; do
    if python -c "import $pkg" &>/dev/null; then
        echo "✅ $pkg already installed, skip."
    else
        echo "⬇️ Installing $pkg..."
        $PIP install "$pkg" \
            --trusted-host pypi.org \
            --trusted-host pypi.python.org \
            --trusted-host files.pythonhosted.org
    fi
done

OPENFOLD_DIR="/GenSIvePFS/users/lutianyu/lf/utils/dplm_utils/dplm/vendor/openfold"
if [ -d "$OPENFOLD_DIR" ]; then
    if python -c "import openfold" &>/dev/null; then
        echo "✅ openfold already installed, skip."
    else
        echo "⬇️ Installing local openfold..."
        $PIP install -e "$OPENFOLD_DIR" \
            --trusted-host pypi.org \
            --trusted-host pypi.python.org \
            --trusted-host files.pythonhosted.org
    fi
else
    echo "⚠️ openfold path not found: $OPENFOLD_DIR"
fi

echo "=== Runnig task ==="
cd /GenSIvePFS/users/lutianyu/lf
conda run -n qwen3 torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPU_COUNT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$SLURM_NODEID \
    pipe.py --config-name='folding'

echo "=== torchrun command executed ==="
EOF
)

echo "=== Cleaning previous builds ==="
rm -r /home/projects/protein/lutianyu/lf/utils/dplm_utils/dplm/vendor/openfold/build && \
rm -r /home/projects/protein/lutianyu/lf/utils/dplm_utils/dplm/vendor/openfold/openfold.egg-info && \
rm /home/projects/protein/lutianyu/lf/utils/dplm_utils/dplm/vendor/openfold/attn_core_inplace_cuda.cpython-310-x86_64-linux-gnu.so
echo "================================"


# Check and create container if not exists on each node(requires few resources, nxpxc = nx1x1):
echo "=== Checking and creating container [$CONTAINER_NAME] on all nodes ==="
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=1 bash -c "
if ! enroot list | grep -q '^$CONTAINER_NAME\$'; then
    echo \"[ \$(hostname) ] Container '$CONTAINER_NAME' not found. Creating...\"
    enroot create --name $CONTAINER_NAME $CONTAINER_PATH
else
    echo \"[ \$(hostname) ] Container '$CONTAINER_NAME' already exists.\"
fi
"

# Export variable in the container:
numactl --cpunodebind=0 --membind=0 srun enroot start -r \
    --mount /home/projects/protein/lutianyu:/GenSIvePFS/users/lutianyu \
    --mount /home/projects/protein/zhangzhe/protenix_data/mmcif:/GenSIvePFS/users/lutianyu/lf/data/raw/rcsb \
    -w $CONTAINER_NAME \
    -- /bin/bash -c "
    export SLURM_JOB_ID=$SLURM_JOB_ID; \
    export SLURM_NNODES=$SLURM_NNODES; \
    export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export SLURM_NODEID=$SLURM_NODEID; \
    export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME; \
    export NCCL_IB_HCA=$NCCL_IB_HCA; \
    export NCCL_DEBUG=$NCCL_DEBUG; \
    export NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS; \
    export TORCH_DISTRIBUTED_TIMEOUT=$TORCH_DISTRIBUTED_TIMEOUT; \
    $TRAIN_SCRIPT"

echo "=== Job completed ==="