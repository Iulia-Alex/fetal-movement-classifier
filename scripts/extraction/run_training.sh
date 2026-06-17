#!/bin/bash
# Train an fECG extraction model.
#
# Usage (local):
#   bash scripts/extraction/run_training.sh v17
#
# Usage (SLURM):
#   sbatch --job-name=train_v17 scripts/extraction/run_training.sh v17
#
# VERSION selects the training script:
#   v1                  → src/extraction/train_movement.py          (Baseline, 128×400)
#   v5 … v13           → src/extraction/train_movement_v{N}.py     (intermediate experiments)
#   v15                 → src/extraction/train_movement_v15.py      (Robust, 128×128)
#   v16                 → src/extraction/train_movement_v16.py      (Attention-gated)
#   v17                 → src/extraction/train_movement_v17.py      (Attention-supervised)
#   v1_resume           → src/extraction/train_movement_v1_resume.py
#
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%x.log

VERSION=${1:-v1}

# ── Environment (update these paths for your cluster) ─────────────────────────
# Update the path below to match your conda environment:
export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH
# Update the conda path below:
source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$VERSION" in
    v1)        SCRIPT="src/extraction/v01_baseline/train_movement.py" ;;
    v1_resume) SCRIPT="src/extraction/v01_baseline/train_movement_v1_resume.py" ;;
    v5|v6)     SCRIPT="src/extraction/v05_v06_amplitude_weighted/train_movement_${VERSION}.py" ;;
    v7|v8)     SCRIPT="src/extraction/v07_v08_direct_prediction/train_movement_${VERSION}.py" ;;
    v9)        SCRIPT="src/extraction/v09_complex_mse/train_movement_v9.py" ;;
    v10|v11)   SCRIPT="src/extraction/v10_v11_peak_supervised/train_movement_${VERSION}.py" ;;
    v12|v13)   SCRIPT="src/extraction/v12_v13_gain_mask/train_movement_${VERSION}.py" ;;
    v14)       SCRIPT="src/extraction/v14_paper_arch_512/train_movement_v14.py" ;;
    v15)       SCRIPT="src/extraction/v15_robust_500hz/train_movement_v15.py" ;;
    v16|v17)   SCRIPT="src/extraction/v16_v17_attention_supervised/train_movement_${VERSION}.py" ;;
    v18)       SCRIPT="src/extraction/v18_attention_1khz/train_movement_v18.py" ;;
    *)         echo "Unknown version: $VERSION" >&2; exit 1 ;;
esac

if [ ! -f "$REPO_ROOT/$SCRIPT" ]; then
    echo "Error: $SCRIPT not found" >&2
    exit 1
fi

cd "$REPO_ROOT"
echo "Training extraction model $VERSION ($SCRIPT)"
python -u "$SCRIPT"
