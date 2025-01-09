#!/bin/bash

SEED=${1:-0}

# image
python main.py --seed $SEED --experiment concept --epochs 1000 --m mlp --data-type image --save

# bitsting
python main.py --seed $SEED --experiment concept --epochs 1000 --m mlp --data-type bits --save