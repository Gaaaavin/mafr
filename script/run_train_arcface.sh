#!/bin/bash

# Experiment name
NAME=arcface_cfp
# Epochs
EPOCH=100
# Learning rate
LR=0.0002
# Training dataset
TRAIN=../data/WebFace1k
# Evaluation dataset
EVAL=../data/cfp
# Log interval
LOG=1
# Batch size
BS=64
# Loss function
LOSS=arc_dist
# Backbone
BACKBONE=resnet50
# Mask portion
MSK=0.5

CUDA_VISIBLE_DEVICES=0 python train_arcface.py --amp --name $NAME -E $EPOCH -t $TRAIN -e $EVAL --log_interval $LOG -B $BS --lr $LR --loss $LOSS --backbone $BACKBONE --msk $MSK