#!/bin/bash

SEED=${1:-0}

# image
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 20 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 20 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 20 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 20 --adaptation-steps 5 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 50 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 50 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 50 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 50 --adaptation-steps 5 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 100 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 100 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 100 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type image --n-samples-per-task 100 --adaptation-steps 5 --skip 2 --save

# bitsting
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 20 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 20 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 20 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 20 --adaptation-steps 5 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 50 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 50 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 50 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 50 --adaptation-steps 5 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 100 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 100 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 100 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type bits --n-samples-per-task 100 --adaptation-steps 5 --skip 2 --save

# number
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 20 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 20 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 20 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 20 --adaptation-steps 5 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 50 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 50 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 50 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 50 --adaptation-steps 5 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 100 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 100 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 100 --adaptation-steps 5 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m mlp --data-type number --n-samples-per-task 100 --adaptation-steps 5 --skip 2 --save
