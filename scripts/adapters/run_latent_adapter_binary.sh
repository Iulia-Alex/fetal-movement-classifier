#!/bin/bash
#SBATCH --output=logs/latent_adapter_binary.log

# Update the path below to match your conda environment:
export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

# Update the conda path below:
source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Train and evaluate binary latent adapter (M14, AP metric)
# Usage: bash run_latent_adapter_binary.sh [v1]   <- extraction variant
VARIANT=${1:-v1}
python -u src/adapters/latent_adapter_binary.py "$VARIANT"
