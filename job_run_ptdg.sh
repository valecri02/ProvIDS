#!/bin/bash
### LSF options
#BSUB -q gpul40s
#BSUB -J emb_size
#BSUB -o tgn_base_%J.out
#BSUB -e tgn_base_%J.err
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_GPUS=1
export NUM_CPUS=6
export NUM_GPUS_PER_TASK=1
export NUM_CPUS_PER_TASK=6

SPLIT="reduced_005"
SAVE_DIR=$(pwd)

SAVE_DIR_BASE="$(pwd)/anomaly_detect"
PRED_DIR="$(pwd)/TGN/ckpt"
GROUND_TRUTH="/work3/s253892/ProvIDS/darpa_labelling/groundtruth"

python -u /work3/s253892/ProvIDS/src/main.py \
  --data_name "darpa_theia_$SPLIT" \
  --model "TGN" \
  --batch 512 \
  --epochs 100 \
  --patience 5 \
  --num_runs 1 \
  --save_dir ${SAVE_DIR} \
  --data_dir "/work3/s253892/ProvIDS/DATA/DATA" \
  --metric auc \
  --verbose \
  --memory false \
  --num_layers 2

SPLIT="005"
N_TOT=$(find "$PRED_DIR" -maxdepth 1 -type f -name "split_conf_*" | wc -l)
N_SEED=$(find "$PRED_DIR" -maxdepth 1 -type f -name "split_conf_0_*" | wc -l)
N_CONF=$(( N_TOT / N_SEED ))


for (( i=0; i < $N_CONF; i++ ))
do
    SAVE_DIR_ANOM="$SAVE_DIR_BASE/conf_${i}"

    mkdir -p "$SAVE_DIR_ANOM"

    python -u /work3/s253892/ProvIDS/src/anomaly_detection.py \
        --prediction_folder "$PRED_DIR" \
        --ground_truth_path "$GROUND_TRUTH" \
        --save_folder "$SAVE_DIR_ANOM" \
        --model_name TGN \
        --dataset THEIA \
        --conf_id "$i" \
        --split "$SPLIT" \
        --num_seeds "$N_SEED"
done

python /work3/s253892/ProvIDS/tools/plot_training_curves.py \
    --ckpt "$PRED_DIR" \
    --out "$SAVE_DIR/training_curves.png" \
    --title "TGAT 15L"

echo "Job finished at $(date)"