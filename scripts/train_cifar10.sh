#!/bin/bash
#SBATCH --account=def-awolson
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --job-name=cifar10
#SBATCH --output=/scratch/%u/adaptovision-replication/outputs/slurm/cifar10_%j.out
#SBATCH --error=/scratch/%u/adaptovision-replication/outputs/slurm/cifar10_%j.err

set -e

module --force purge
module load StdEnv/2023
module load python/3.11

cd /scratch/$USER/adaptovision-replication

mkdir -p outputs/slurm
mkdir -p outputs/runs

source .venv/bin/activate

python -m adaptovision.modeling.train \
  --config configs/cifar10.yaml
