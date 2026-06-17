#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=logs/waveform_adapter_train.log

# SUPERSEDED by the latent adapter.
# Trains the beat-level waveform adapter on DB_1 (regression on R-peak amplitudes,
# re-injected as gain into the decoded signal before M15 classification).

# Update the path below to match your conda environment:
export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

# Update the conda path below:
source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python -u src/latent_adapter/train_adapter_db1.py
