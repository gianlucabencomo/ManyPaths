#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=1 slurm/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=2 slurm/cnn.slurm;