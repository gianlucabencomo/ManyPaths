#!/bin/bash

SEED=${1:-0}

# image
python main.py --seed $SEED --epochs 1000 --m mlp --data-type image --skip 1 --save
python main.py --seed $SEED --epochs 1000 --m mlp --data-type image --skip 2 --save

# bitsting
python main.py --seed $SEED --epochs 1000 --m mlp --data-type bits --skip 1 --save
python main.py --seed $SEED --epochs 1000 --m mlp --data-type bits --skip 2 --save

# number
python main.py --seed $SEED --epochs 1000 --m mlp --data-type number --skip 1 --save
python main.py --seed $SEED --epochs 1000 --m mlp --data-type number --skip 2 --save
