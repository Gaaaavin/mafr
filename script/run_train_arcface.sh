#!/bin/bash

# Experiment name
NAME=ArcFace
# Epochs
EPOCH=200
# Learning rate
LR=5e-4
# Training dataset
TRAIN=../data/WebFace1k
# Masked training dataset
MASK=../data/WebFace1k_msk
# Evaluation dataset
EVAL=../data/WebFaceEval
# Log interval
LOG=1
# Batch size
BS=512
# Pretrained
pretraind=False

CUDA_VISIBLE_DEVICES=0 python train_arcface.py --name $NAME -E $EPOCH -t $TRAIN -m $MASK -e $EVAL --log_interval $LOG -B $BS --lr $LR