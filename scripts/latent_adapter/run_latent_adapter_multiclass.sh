#!/bin/bash
#SBATCH --output=logs/latent_adapter_multiclass.log

# Update the path below to match your conda environment:
export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

# Update the conda path below:
source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Evaluate latent adapter on 4-class task (M15, recovers no-move + helix)
python -u src/latent_adapter/latent_adapter_multiclass.py
