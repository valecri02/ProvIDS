#!/bin/bash
### LSF options
#BSUB -q gpuv100
#BSUB -J anomaly
#BSUB -o anomaly_detect_%J.out
#BSUB -e anomaly_detect_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

set -euo pipefail

module purge

# Conda setup
source ~/miniforge3/bin/activate
conda activate ctdg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export NUM_CPUS=4
export NUM_GPUS=1


PRED_DIR=$(pwd)/TGN/ckpt
SAVE_DIR=$(pwd)/anomaly_detect
CONF=0

cd ${PRED_DIR}
N_SEED=$(find . -maxdepth 1 -type f -name "split_conf_${CONF}_*" | wc -l)

# Repo root
cd "/work3/s253892/ProvIDS"
echo "Repo CWD: $(pwd)"



python -c "import torch, pandas, numpy, scipy, sklearn; import torch_geometric; print('imports ok')"

# Run training (from README)
cd src
python -u anomaly_detection.py \
    --prediction_folder ${PRED_DIR} \
    --ground_truth_path /work3/s253892/ProvIDS/darpa_labelling/groundtruth \
    --save_folder ${SAVE_DIR} \
    --model_name TGN \
    --dataset THEIA \
    --conf_id ${CONF} \
    --wandb \
    --num_seeds ${N_SEED}

echo "Job finished at $(date)"