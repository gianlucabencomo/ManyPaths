#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=1 slurm/lstm.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=2 slurm/lstm.slurm;

# bitsting
sbatch --export=SEED=$SEED,DATA_TYPE=bits,SKIP=1 slurm/lstm.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=bits,SKIP=2 slurm/lstm.slurm;