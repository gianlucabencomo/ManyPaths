#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,ALPHABET=all slurm/omniglot/lstm.slurm;
sbatch --export=SEED=$SEED,ALPHABET=asian slurm/omniglot/lstm.slurm;
sbatch --export=SEED=$SEED,ALPHABET=ancient slurm/omniglot/lstm.slurm;
sbatch --export=SEED=$SEED,ALPHABET=european slurm/omniglot/lstm.slurm;
sbatch --export=SEED=$SEED,ALPHABET=middle slurm/omniglot/lstm.slurm;