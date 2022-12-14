#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=focus
#SBATCH --mail-type=END,FAILED
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=focus_%j.out
#SBATCH --error=focus_%j.err
#SBATCH --gres=gpu:a100:1

# Experiment name
NAME=FocusFace
# Epochs
EPOCH=500
# Learning rate
LR=1e-4
# Training dataset
TRAIN=/WebFace1k
# Evaluation dataset
EVAL=/WebFaceEval
# Log interval
LOG=1
# Batch size
BS=256


module purge
cd /scratch/xl3136/mafr/script
singularity exec --nv \
    --overlay /scratch/xl3136/environments/focusface.ext3:ro \
    --overlay /scratch/xl3136/mafr/data/WebFace1k.sqf \
    --overlay /scratch/xl3136/mafr/data/WebFaceEval.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; 
    python train_focusface.py --amp --name $NAME -E $EPOCH -t $TRAIN -e $EVAL --log_interval $LOG -B $BS"