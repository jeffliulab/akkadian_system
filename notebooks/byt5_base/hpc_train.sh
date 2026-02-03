#!/bin/bash
#SBATCH -J byt5_akkadian
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --output=byt5_train_%j.out
#SBATCH --error=byt5_train_%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@tufts.edu

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# 加载模块
module load miniforge/25.3.0

# 确认清单
# 1、工作目录
# 2、工作环境
# 3、工作环境依赖
# 4、缓存目录

# 工作目录
WORK_DIR=/cluster/tufts/c26sp1ee0141/pliu07/akadian-base
cd $WORK_DIR

echo "Working directory: $(pwd)"

# Conda环境路径
ENV_PATH=/cluster/tufts/c26sp1ee0141/pliu07/byt5_base_env

# 检查环境是否存在，不存在则创建
if [ ! -d "$ENV_PATH" ]; then
    echo "创建conda环境: $ENV_PATH"
    conda create -p $ENV_PATH python=3.10 -y
    conda activate $ENV_PATH
    echo "安装依赖包..."
    pip install torch transformers datasets sacrebleu accelerate tensorboard pandas numpy tqdm evaluate
else
    echo "激活已有环境: $ENV_PATH"
    conda activate $ENV_PATH
fi

# 设置缓存目录
export HF_HOME=/cluster/tufts/c26sp1ee0141/pliu07/huggingface_cache
export TRANSFORMERS_CACHE=/cluster/tufts/c26sp1ee0141/pliu07/huggingface_cache
mkdir -p $HF_HOME

# 优化PyTorch显存分配（避免碎片化）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "环境配置完成"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

# 开始训练
echo "开始训练..."
python 2_train_hpc.py

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="