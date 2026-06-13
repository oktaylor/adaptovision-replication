#!/bin/bash
set -e

module --force purge
module load StdEnv/2023
module load python/3.11

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the remaining Python dependencies from PyPI.
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Environment created successfully."
