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
# module load python3/3.12.9

# Conda setup
source ~/miniforge3/bin/activate
conda activate ctdg_pyg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export NUM_CPUS=4
export NUM_GPUS=1

# Repo root
cd "$HOME/other_repos/ProvCTDG"
echo "Repo CWD: $(pwd)"

DATA_DIR="$HOME/other_repos/ProvCTDG/data/DATA"
DATA_NAME="darpa_theia_0to25"
SAVE_DIR="$HOME/other_repos/ProvCTDG/experiments/tgn_mem_theia_1run"
MODEL="TGN"

python -c "import torch, pandas, numpy, scipy, sklearn; import torch_geometric; print('imports ok')"

# Run training (from README)
cd src
python -u main.py \
  --data_name ${DATA_NAME} \
  --model ${MODEL} \
  --version temporal \
  --parallelism 1 \
  --epochs 50 \
  --batch 200 \
  --save_dir ${SAVE_DIR} \
  --data_dir ${DATA_DIR} \
  --num_runs 1 \
  --patience 5 \
  --neg_sampler HeterogeneousNegativeSampler \
  --metric auc \
  --strategy split \
  --exp_seed 9

echo "Job finished at $(date)"