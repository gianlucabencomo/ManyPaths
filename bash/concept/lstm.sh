#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image slurm/concept/lstm.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image slurm/concept/lstm.slurm;

# bitsting
sbatch --export=SEED=$SEED,DATA_TYPE=bits slurm/concept/lstm.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=bits slurm/concept/lstm.slurm;