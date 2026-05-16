#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
THESIS_DIR="$(cd "$PROVIDS_DIR/.." && pwd)"

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate ctdg_pyg

export KMP_DUPLICATE_LIB_OK=TRUE
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR=/private/tmp/mplconfig

DATA_NAME="${1:-darpa_tgrab_periodicity_200n}"
MODEL="${2:-TGN}"
SAVE_DIR="${3:-$PROVIDS_DIR/RESULTS/tgrab_local_${MODEL}}"

rm -rf "$PROVIDS_DIR/DATA/$DATA_NAME/temporal_processed"

cd "$SCRIPT_DIR"

python main.py \
  --data_dir "$PROVIDS_DIR/DATA" \
  --data_name "$DATA_NAME" \
  --version temporal \
  --model "$MODEL" \
  --neg_sampler NegativeSampler \
  --cpu \
  --debug \
  --num_runs 1 \
  --epochs 100 \
  --patience 5 \
  --batch 50 \
  --save_dir "$SAVE_DIR" \
  --overwrite_ckpt
