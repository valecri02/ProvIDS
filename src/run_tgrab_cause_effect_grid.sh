#!/bin/bash
### LSF options
#BSUB -q gpul40s
#BSUB -J emb_size
#BSUB -o tgn_base.out
#BSUB -e tgn_base.err
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -M 5GB
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

set -euo pipefail

module purge

module load cuda/11.7

# Conda setup
source ~/miniforge3/bin/activate
conda activate ctdg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$PROVIDS_DIR/DATA}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-$PROVIDS_DIR/RESULTS/tgrab_cause_effect_grid/TGN_noMem_1L}"

MEMORY="${MEMORY:-false}"

LAGS=(${LAGS:-1 4 16 64 256 1024})

cd "$SCRIPT_DIR"
mkdir -p "$SAVE_DIR_BASE"

for LAG in "${LAGS[@]}"; do
  DATA_NAME="darpa_tgrab_cause_effect_lag${LAG}_101n"
  SAVE_DIR="$SAVE_DIR_BASE/${DATA_NAME}_${MODEL}_memory-${MEMORY}"

  rm -rf "$DATA_DIR/$DATA_NAME"/temporal_processed*

  python -u main.py \
    --verbose \
    --data_dir "$DATA_DIR" \
    --data_name "$DATA_NAME" \
    --version temporal \
    --save_dir "$SAVE_DIR" \
    --model "TGN" \
    --neg_sampler NegativeSampler \
    --parallelism 3 \
    --num_runs 3 \
    --epochs 1000 \
    --patience 10 \
    --batch 512 \
    --metric ap \
    --memory "$MEMORY" \
    --num_layers 1
done
