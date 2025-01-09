#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=1 slurm/transformer.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=image,SKIP=2 slurm/transformer.slurm;

# bitsting
sbatch --export=SEED=$SEED,DATA_TYPE=bits,SKIP=1 slurm/transformer.slurm;
sbatch --export=SEED=$SEED,DATA_TYPE=bits,SKIP=2 slurm/transformer.slurm;