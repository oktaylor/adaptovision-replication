#!/bin/bash
#SBATCH --account=def-awolson
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --job-name=check_cuda
#SBATCH --output=/scratch/%u/adaptovision-replication/outputs/logs/check_cuda_%j.out
#SBATCH --error=/scratch/%u/adaptovision-replication/outputs/logs/check_cuda_%j.err

set -e

module --force purge
module load StdEnv/2023
module load python/3.11

cd /scratch/$USER/adaptovision-replication
mkdir -p outputs/logs

source .venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("gpu:", torch.cuda.get_device_name(0))
PY
