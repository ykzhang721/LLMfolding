#!/bin/bash
#SBATCH --account=protein
#SBATCH --partition=AISS2024110101
#SBATCH --job-name=lf-ray
#SBATCH --output=ray.log
#SBATCH --error=ray.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=2000
#SBATCH --export=ALL
#SBATCH -w cp2-dgx-[010,013]

# ------------------ 配置 ------------------
CONTAINER_PATH=/home/projects/protein/lutianyu/images/modern.sqsh
CONTAINER_NAME=modern
PYTHON_BIN=/root/miniconda3/envs/qwen3/bin/python
RAY_BIN=/root/miniconda3/envs/qwen3/bin/ray

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=28
export TORCH_DISTRIBUTED_TIMEOUT=1800

# ------------------ 节点解析 ------------------
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
port=6379
echo "Head node: $head_node ($head_ip)"
echo "All nodes: ${nodes[@]}"

# ------------------ 启动容器（每节点） ------------------
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
if ! enroot list | grep -q '^$CONTAINER_NAME\$'; then
    echo \"Creating container on \$(hostname)...\"
    enroot create --name $CONTAINER_NAME $CONTAINER_PATH
else
    echo \"Container already exists on \$(hostname)\"
fi
"

# ------------------ 启动 Ray Head ------------------
echo "Starting Ray head on $head_node"
srun --nodes=1 --ntasks=1 -w $head_node enroot start -r \
    --mount /home/projects/protein/lutianyu:/GenSIvePFS/users/lutianyu \
    -w $CONTAINER_NAME \
    -- bash -c "
        $RAY_BIN stop >/dev/null 2>&1;
        $RAY_BIN start --head --node-ip-address=$head_ip --port=$port \
            --logging-dir=/GenSIvePFS/users/lutianyu/lf/ray_head.log \
            --num-cpus=224 --num-gpus=8 --dashboard-port=8265 > /tmp/ray_head.log 2>&1 &
        sleep 10
    "

# ------------------ 启动 Ray Workers ------------------
for node in "${nodes[@]:1}"; do
    echo "Starting Ray worker on $node"
    srun --nodes=1 --ntasks=1 -w $node enroot start -r \
        --mount /home/projects/protein/lutianyu:/GenSIvePFS/users/lutianyu \
        -w $CONTAINER_NAME \
        -- bash -c "
            $RAY_BIN stop >/dev/null 2>&1;
            $RAY_BIN start --address="$head_ip:$port" \
            --logging-dir=/GenSIvePFS/users/lutianyu/lf/ray_worker.log \
            --num-cpus=224 --num-gpus=8 > /tmp/ray_worker.log 2>&1 &
            sleep 5
        "
done

echo "Waiting 20s for all workers to join..."
sleep 20

# ------------------ 执行任务 (仅在 head 节点) ------------------
echo "Running Ray job on head node..."
srun --nodes=1 --ntasks=1 -w $head_node enroot start -r \
    --mount /home/projects/protein/lutianyu:/GenSIvePFS/users/lutianyu \
    -w $CONTAINER_NAME \
    -- bash -c "
        cd /GenSIvePFS/users/lutianyu/lf;
        export RAY_DEDUP_LOGS=0;
        export RAY_ADDRESS="$head_ip:$port";
        echo '=== Job Started ===';
        conda run -n qwen3 $PYTHON_BIN t.py;
        echo '=== Job Finished ===';
    "

# ------------------ 清理 ------------------
echo "Cleaning up Ray processes..."
for node in "${nodes[@]}"; do
    srun --nodes=1 --ntasks=1 -w $node enroot start -r -w $CONTAINER_NAME \
        -- bash -c "$RAY_BIN stop" &
done
wait
echo "=== All Ray processes stopped ==="