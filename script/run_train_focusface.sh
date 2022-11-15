#!/bin/bash

# Experiment name
NAME=FocusFace
# Epochs
EPOCH=500
# Learning rate
LR=1e-4
# Training dataset
TRAIN=../data/WebFace1k
# Evaluation dataset
EVAL=../data/WebFaceEval
# Log interval
LOG=1
# Batch size
BS=96

CUDA_VISIBLE_DEVICES=0 python train_focusface.py --amp -v --name $NAME -E $EPOCH --lr $LR -t $TRAIN -e $EVAL --log_interval $LOG -B $BS