#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image,N_SAMPLES=20,STEPS=1,SKIP=1 slurm/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,N_SAMPLES=20,STEPS=1,SKIP=2 slurm/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,N_SAMPLES=50,STEPS=1,SKIP=1 slurm/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,N_SAMPLES=50,STEPS=1,SKIP=2 slurm/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,N_SAMPLES=100,STEPS=1,SKIP=1 slurm/cnn.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,N_SAMPLES=100,STEPS=1,SKIP=2 slurm/cnn.slurm;