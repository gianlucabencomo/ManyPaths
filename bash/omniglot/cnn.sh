#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,ALPHABET=all slurm/omniglot/cnn.slurm;
sbatch --export=SEED=$SEED,ALPHABET=asian slurm/omniglot/cnn.slurm;
sbatch --export=SEED=$SEED,ALPHABET=ancient slurm/omniglot/cnn.slurm;
sbatch --export=SEED=$SEED,ALPHABET=european slurm/omniglot/cnn.slurm;
sbatch --export=SEED=$SEED,ALPHABET=middle slurm/omniglot/cnn.slurm;