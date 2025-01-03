#!/bin/bash

SEED=${1:-0}

# image
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 20 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 20 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 20 --adaptation-steps 10 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 20 --adaptation-steps 10 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 50 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 50 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 50 --adaptation-steps 10 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 50 --adaptation-steps 10 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 100 --adaptation-steps 1 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 100 --adaptation-steps 1 --skip 2 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 100 --adaptation-steps 10 --skip 1 --save
python train.py --seed $SEED --epochs 1000 --m cnn --data-type image --n-samples-per-task 100 --adaptation-steps 10 --skip 2 --save