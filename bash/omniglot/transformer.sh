#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,ALPHABET=all slurm/omniglot/transformer.slurm;
sbatch --export=SEED=$SEED,ALPHABET=asian slurm/omniglot/transformer.slurm;
sbatch --export=SEED=$SEED,ALPHABET=ancient slurm/omniglot/transformer.slurm;
sbatch --export=SEED=$SEED,ALPHABET=european slurm/omniglot/transformer.slurm;
sbatch --export=SEED=$SEED,ALPHABET=middle slurm/omniglot/transformer.slurm;