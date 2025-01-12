#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,ALPHABET=all slurm/omniglot/mlp.slurm;
sbatch --export=SEED=$SEED,ALPHABET=asian slurm/omniglot/mlp.slurm;
sbatch --export=SEED=$SEED,ALPHABET=ancient slurm/omniglot/mlp.slurm;
sbatch --export=SEED=$SEED,ALPHABET=european slurm/omniglot/mlp.slurm;
sbatch --export=SEED=$SEED,ALPHABET=middle slurm/omniglot/mlp.slurm;