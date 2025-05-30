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
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} tools/train.py --launcher pytorch ${PY_ARGS}

#arbe
#bash tools/scripts/dist_train.sh 2 --cfg_file tools/cfgs/dual_radar_models/pointpillar_arbe.yaml
