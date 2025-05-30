#!/usr/bin/env bash

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 设置主节点
# 节点列表
NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
# 对第一个节点赋值为主节点
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
# 计数器
NODE_COUNT=0
# 一共的节点数
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))

# 打印
echo $SLURM_NODEID
echo $NODELIST
echo $MASTER_NODE
echo $NODE_NUM

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
PY_ARGS=${@:4}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py --launcher slurm --tcp_port $PORT ${PY_ARGS}
