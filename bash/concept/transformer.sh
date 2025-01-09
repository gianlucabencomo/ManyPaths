#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image slurm/concept/transformer.slurm;

# bitsting
sbatch --export=SEED=$SEED,DATA_TYPE=bits slurm/concept/transformer.slurm;