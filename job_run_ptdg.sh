#!/bin/bash
### LSF options
#BSUB -q gpuv100
#BSUB -J tgn_mem_th
#BSUB -o tgn_mem_theia_%J.out
#BSUB -e tgn_mem_theia_%J.err
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

SAVE_DIR=$(pwd)

# Repo root
cd "/work3/s253892/ProvIDS"
echo "Repo CWD: $(pwd)"

python -c "import torch, pandas, numpy, scipy, sklearn; import torch_geometric; print('imports ok')"

# Run training (from README)
cd src
python -u main.py \
  --data_name "darpa_theia_0to25" \
  --model "TGN" \
  --parallelism 2 \
  --batch 200 \
  --epochs 50 \
  --patience 5 \
  --save_dir ${SAVE_DIR} \
  --data_dir "/work3/s253892/ProvIDS/DATA/DATA" \
  --num_runs 1 \
  --metric auc \
  --memory_enhancement 1

echo "Job finished at $(date)"