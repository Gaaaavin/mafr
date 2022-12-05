#!/bin/bash

# Experiment name
NAME=ArcFace
# Epochs
EPOCH=100
# Learning rate
LR=0.0005
# Training dataset
TRAIN=../data/WebFace1k
# Evaluation dataset
EVAL=../data/WebFace1k
# Log interval
LOG=1
# Batch size
BS=512
# Loss function
LOSS=arc_dist

CUDA_VISIBLE_DEVICES=0 python train_arcface.py --amp --name $NAME -E $EPOCH -t $TRAIN -e $EVAL --log_interval $LOG -B $BS --lr $LR --loss $LOSS