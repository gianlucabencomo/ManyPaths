#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=gb5435@princeton.edu

module purge
module load anaconda3/2024.6
module load cudatoolkit/12.4
conda activate paths

SEED=${SEED:-0}                     # Default seed is 0
ALPHABET=${ALPHABET:-all}

python main.py --seed $SEED --experiment omniglot --epochs 1000 --m cnn --a $ALPHABET --save