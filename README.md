# Fetal Movement Classifier

Code for the master's thesis project called 'Methods for Automatic Analysis of the Fetal Electrocardiogram Based on Deep Learning for Assessing Fetal Health Status'.

The project is a **two-stage pipeline**:
1. **fECG extraction** — complex-valued U-Net family (v1–v17) that separates the fetal ECG from a 6-channel abdominal mixture
2. **Downstream analysis** of the extracted signal, covering four approaches:
   - **Latent adapter** *(main)* — bottleneck correction inserted between the frozen extractor and the frozen classifier, trained label-only on DB_1 (~130 k params, 2× Conv1×1 + GroupNorm + ReLU, residual/zero-init)
   - **Waveform adapter** *(superseded)* — beat-level amplitude correction on the decoded signal before classification; less stable and superseded by the latent adapter
   - **Fine-tuning** — full re-training of the binary classifier on extracted signals (domain adaptation; not an adapter)
   - **Cross-channel classifier** — random-forest model using all six leads jointly (spectral + trajectory features), trained on extracted signals

The downstream classifier (1-D Attention U-Net, M14/M15/M18) is a **pre-existing model** (collaborator contribution), provided in `src/shared/pretrained_clf.py`. The thesis contribution is the coupling experiment, the diagnostics, and the four downstream approaches above.

---

## Repository layout

```
src/                              Python source (entry scripts add src/ to sys.path via pathsetup.py)
├── config.py                     Cluster paths, overridable via FECG_ROOT / FECG_DATA_ROOT
├── pathsetup.py                  Registers every src/ subfolder on sys.path (flat import names)
│
├── extraction/                   fECG extraction — complex-valued U-Net family
│   ├── pipeline_registry.py      version tag → architecture + checkpoint path + infer fn
│   ├── infer_testdb.py           Batch inference on Test-DB for any registered version
│   ├── extract_save_npy.py       Cache extracted signals to .npy for downstream tasks
│   ├── eval_qrs_f1{,_v2,_v3}.py  QRS-peak F1 on Test-DB (3 iterations; v3 = final, 500 Hz)
│   ├── smoke_pipeline.py         Quick sanity check: load → infer → F1 on one file
│   └── v01_baseline/ … v18_attention_500hz/
│                                 per-experiment folders, each with complex_network_vN.py,
│                                 movement_dataset_vN.py and train_movement_vN.py
│                                 (v01, v05_v06, v07_v08, v09, v10_v11, v12_v13, v14, v15, v16_v17, v18)
│
├── shared/                       shared utilities
│   ├── pretrained_clf.py         AttUNet1D M14 (binary) / M15 (4-class) / M18 — pre-existing, frozen; + features
│   ├── diag_info_ceiling.py      Load signals/masks, detect QRS peaks, amplitude-correlation ceiling
│   └── pipeline_clf.py           Pipeline helpers for classifier evaluation
│
├── latent_adapter/               adapter family: frozen backbone + small learned correction
│   ├── latent_adapter.py             LatentAdapter class + multiclass train on DB_1 (label-only)
│   ├── latent_adapter_binary.py      Binary adapter (M14), trained/eval on extracted (113 signals)
│   ├── latent_adapter_multiclass.py  4-class adapter (M15): recovers no-move + helix
│   ├── eval_adapter_binary_prauc.py  PR-AUC + best-F1 metrics helper
│   ├── eval_binary_adapter.py        Comparison table: base / adapter / GT ceiling
│   ├── build_adapter.py              (superseded) beat-level MLP adapter + POC evaluation
│   └── train_adapter_db1.py          (superseded) train waveform adapter on DB_1
│
├── clf_finetuning/               full re-training of the pre-trained classifier (non-adapter)
│   ├── finetune_clf_extracted_binary.py  Re-train M14 end-to-end on extracted signals
│   └── fuse_channels_binary.py           Channel-fusion head on top of the fine-tuned M14
│
├── crosschannel_rf/              random forest on hand-crafted cross-channel features
│   ├── probe_crosschannel_multiclass.py  Separability probes + cross-channel feature definitions
│   ├── dense_multiclass_crosschannel.py  Dense per-sample RF, sliding window
│   ├── two_stage_dense.py                Two-stage: binary detection → type
│   ├── two_stage_dense_improved.py       Two-stage with a shorter detection window + threshold calibration
│   ├── two_stage_improved.py             Two-stage + spatial-coherence features + balanced threshold
│   ├── two_stage_3class.py               3-class variant {no-move, directed (linear+spline), helix}
│   └── final_two_stage_eval.py           Final 5-fold CV evaluation
│
└── eval/                         diagnostics & statistics (examiner-driven analyses)
    ├── correction_stats.py             Bootstrap CI + paired Wilcoxon on the binary correction gains
    ├── control_perlead_4class.py       Per-lead 4-class control (domain-matched)
    ├── control_perchannel_4class.py    Representation decomposition control (spectral / geom / full)
    ├── disentangle_amp_corr.py         Table 4.8 disambiguation (detected vs reference peaks)
    ├── diag_compare_models.py          Shape-preservation comparison across extraction models
    └── probe_separability.py           Window-level separability probe (linear/spline strategy)

scripts/
├── extraction/
│   └── run_training.sh           Generic SLURM/local launcher — pass version as argument
└── latent_adapter/
    ├── run_latent_adapter_train.sh       Latent adapter — multiclass train
    ├── run_latent_adapter_binary.sh      Latent adapter — binary eval
    ├── run_latent_adapter_multiclass.sh  Latent adapter — 4-class eval
    └── run_waveform_adapter_train.sh     Waveform adapter (superseded)

logs/                             SLURM log output (not tracked)
```

---

## Extraction model progression

The three **thesis models** are referred to by role, not version number:

| Role | Internal | Architecture | Pipeline | Notes |
|---|---|---|---|---|
| **Baseline** | v1 | ComplexUNet (0.59 M) | 128×400 | Soft mask, additive skips, leaky ReLU |
| **Attention-gated** | v16 | ComplexAttentionUNet | 128×128 | 3 cross-attention gates |
| **Attention-supervised** | v17 | ComplexAttentionUNet | 128×128 | v16 + attention loss on fQRS mask |

Intermediate versions (v5–v15) form the design narrative (loss ablation, mask types, architecture changes) and their training scripts are included for reproducibility. All verified models are registered in `pipeline_registry.py`.

---

## External assets (not in this repo)

All cluster paths are centralised in `src/config.py` and configured via two environment variables:

```bash
export FECG_ROOT=/path/to/fECG_approx          # model checkpoints, results, cached signals
export FECG_DATA_ROOT=/path/to/fECG_mvm_DB     # movement ECG database
```

Copy `.env.example` to `.env` and set the two variables before running any script. The derived paths are:

| Constant | Derived path | Contents |
|---|---|---|
| `MODELS_DIR` | `$FECG_ROOT/models` | Extraction model checkpoints (`*.pth`) |
| `CLF_MODELS_DIR` | `$FECG_ROOT/classifier_models` | Pre-trained classifier weights (M14, M15, M18) |
| `RESULTS_DIR` | `$FECG_ROOT/results_fECG_extraction` | Adapter checkpoints, eval JSON |
| `DB1_LONG` | `$FECG_DATA_ROOT/DB_1/Long_time_intervals` | Training `.mat` files for adapter/fine-tuning |
| `NPY` | `$FECG_DATA_ROOT/Final_Test_DB_npy` | Preprocessed `.npy` test signals (122 recordings) |
| `TEST_DIR` | `$FECG_DATA_ROOT/Test_DB` | Clean test recordings `Sem1…Sem11` |

---

## Environment

```bash
conda create -n ecg python=3.10
conda activate ecg
# PyTorch with CUDA 12.1 — adjust index URL for your driver:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.4.1, CUDA 12.1.

---

## How to run

All scripts add the `src/` root to `sys.path` and import `pathsetup`, which registers every subfolder, so the flat import names resolve from any subfolder. Run a script from inside its subfolder, or as `python src/<area>/<script>.py` from the repo root.

### fECG extraction — train

```bash
# Local
bash scripts/extraction/run_training.sh v17

# SLURM
sbatch --job-name=train_v17 scripts/extraction/run_training.sh v17
```

Supported version tags: `v1`, `v5`–`v18`, `v1_resume`.

### fECG extraction — evaluate

```bash
cd src/extraction
python extract_save_npy.py v1          # cache extracted signals
python eval_qrs_f1_v3.py              # QRS F1 on Test-DB (500 Hz corrected)
python infer_testdb.py                 # batch inference for all registered versions
python smoke_pipeline.py               # quick sanity check
```

### Latent adapter (main)

```bash
# Train (multiclass, label-only, ~130 k params, CPU-friendly):
bash scripts/latent_adapter/run_latent_adapter_train.sh

# Binary AP evaluation:
bash scripts/latent_adapter/run_latent_adapter_binary.sh v1

# 4-class evaluation:
bash scripts/latent_adapter/run_latent_adapter_multiclass.sh
```

Or directly from `src/latent_adapter/`:
```bash
cd src/latent_adapter
python latent_adapter.py
python latent_adapter_binary.py v1
python eval_binary_adapter.py
```

### Waveform adapter (superseded)

```bash
bash scripts/latent_adapter/run_waveform_adapter_train.sh
# or: cd src/latent_adapter && python train_adapter_db1.py
```

### Fine-tuning (non-adapter)

```bash
cd src/clf_finetuning
python finetune_clf_extracted_binary.py          # probe mode
python finetune_clf_extracted_binary.py train    # full run
```

### Cross-channel classifier (non-adapter)

```bash
cd src/crosschannel_rf
python probe_crosschannel_multiclass.py   # oracle separability probes
python dense_multiclass_crosschannel.py   # 5-fold CV on extracted signals
python two_stage_improved.py              # two-stage + spatial-coherence features
python two_stage_3class.py                # 3-class variant (no-move / directed / helix)
python final_two_stage_eval.py            # final two-stage 5-fold evaluation
```

### Diagnostics & statistics

```bash
cd src/eval
python correction_stats.py            # bootstrap CI + Wilcoxon on the correction gains
python control_perlead_4class.py      # per-lead 4-class control
python control_perchannel_4class.py   # representation decomposition control
python disentangle_amp_corr.py        # Table 4.8 disambiguation
python diag_compare_models.py         # shape preservation across extraction models
python probe_separability.py          # window-level separability probe
```

---

## Citation / contact

Iulia Orvas — master's thesis, Universitatea Politehnica din București, 2026.
