"""
Cluster path configuration — override any constant via environment variables.

Usage example:
    export FECG_ROOT=/your/path/to/fECG_approx
    export FECG_DATA_ROOT=/your/path/to/fECG_mvm_DB
    python src/extraction/eval_qrs_f1_v3.py
"""
import os

FECG_ROOT      = os.environ.get('FECG_ROOT',      '/shared_storage/iulia.orvas/paper/fECG_approx')
FECG_DATA_ROOT = os.environ.get('FECG_DATA_ROOT', '/shared_storage/stan.edward/fECG_mvm_DB')

# Extraction model checkpoints
MODELS_DIR     = os.path.join(FECG_ROOT, 'models')
# Pre-trained classifier checkpoints (M14, M15, M18)
CLF_MODELS_DIR = os.path.join(FECG_ROOT, 'classifier_models')
# Results directory (adapter checkpoints, eval JSON, cached predictions)
RESULTS_DIR    = os.path.join(FECG_ROOT, 'results_fECG_extraction')
EVAL_DIR       = os.path.join(RESULTS_DIR, 'pipeline_eval')
XLSX           = os.path.join(FECG_ROOT, 'experiments.xlsx')

# Movement DB paths
DB1_LONG = os.path.join(FECG_DATA_ROOT, 'DB_1', 'Long_time_intervals')
NPY      = os.path.join(FECG_DATA_ROOT, 'Final_Test_DB_npy')
TEST_DIR = os.path.join(FECG_DATA_ROOT, 'Test_DB')
