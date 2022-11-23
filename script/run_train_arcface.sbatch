#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=arc
#SBATCH --mail-type=END,FAILED
#SBATCH --mail-user=zs1542@nyu.edu
#SBATCH --output=/scratch/zs1542/MAFR/output_scripts/arc_%j.out
#SBATCH --error=/scratch/zs1542/MAFR/output_scripts/arc_%j.err
#SBATCH --gres=gpu:rtx8000:1



# Experiment name
NAME=ArcFace
# Epochs
EPOCH=100
# Learning rate
LR=1e-4
# Training dataset
TRAIN=/WebFace1k
# Evaluation dataset
EVAL=/WebFaceEval
# Log interval
LOG=1
# Batch size
BS=100
# Pretrained
pretraind=False


module purge
cd /scratch/zs1542/MAFR/script
singularity exec --nv \
    --overlay /scratch/zs1542/my_env/focus_face.ext3:ro \
    --overlay /scratch/zs1542/MAFR/dataset/WebFace1k.sqf \
    --overlay /scratch/zs1542/MAFR/dataset/WebFaceEval.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; 
    CUDA_VISIBLE_DEVICES=0 python train_arcface.py --amp --name $NAME -E $EPOCH -t $TRAIN -e $EVAL --log_interval $LOG -B $BS --pretrained $pretrained"