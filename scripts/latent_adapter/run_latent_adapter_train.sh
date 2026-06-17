#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=logs/latent_adapter_train.log

# Update the path below to match your conda environment:
export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

# Update the conda path below:
source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Train latent adapter (multiclass, 4-class, label-only, ~130k params)
python -u src/latent_adapter/latent_adapter.py
