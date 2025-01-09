#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=1 slurm/mod/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=2 slurm/mod/cnn.slurm;