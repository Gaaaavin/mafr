#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=arcface
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=arc_%j.out
#SBATCH --error=arc_%j.err
#SBATCH --gres=gpu:a100:1

# Experiment name
NAME=ArcFace_nodist
# Epochs
EPOCH=100
# Learning rate
LR=0.0002
# Training dataset
TRAIN=/WebFace30k
# Evaluation dataset
EVAL=/WebFace1k
# Log interval
LOG=1
# Batch size
BS=640
# Loss function
LOSS=arcface
# Backbone
BACKBONE=resnet18


module purge
cd /scratch/xl3136/mafr/script
singularity exec --nv \
    --overlay /scratch/xl3136/environments/mafr.ext3:ro \
    --overlay /scratch/xl3136/mafr/data"$TRAIN".sqf \
    --overlay /scratch/xl3136/mafr/data"$TRAIN"_masked.sqf \
    --overlay /scratch/xl3136/mafr/data"$EVAL".sqf \
    --overlay /scratch/xl3136/mafr/data"$EVAL"_masked.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; 
    python train_arcface.py --amp --name $NAME -E $EPOCH -t $TRAIN -e $EVAL --log_interval $LOG -B $BS --lr $LR --loss $LOSS --backbone $BACKBONE"