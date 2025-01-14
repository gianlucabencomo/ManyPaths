#!/bin/bash
#SBATCH --job-name=lstm_cpu_baseline  # job name
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # moderate parallelization
#SBATCH --mem=10G                    # moderate memory
#SBATCH --time=96:00:00               # expected runtime
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=gb5435@princeton.edu

module purge
module load anaconda3/2024.6
conda activate paths

SEED=${SEED:-0}
ALPHABET=${ALPHABET:-all}

python -u main.py --seed $SEED --experiment omniglot --epochs 1000 --m lstm --a $ALPHABET --save