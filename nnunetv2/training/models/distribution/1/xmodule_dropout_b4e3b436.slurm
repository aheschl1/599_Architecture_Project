#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2

#Environment setup
source ~/software/init-conda
conda activate torch_env
#Commands
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 21 3d_fullres 0 -tr ourTrainer -model /home/andrewheschl/Documents/Seg3D/nnunetv2/training/models/mc_dropout_models/xmodules/0.25/3_11.json &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 21 3d_fullres 0 -tr ourTrainer -model /home/andrewheschl/Documents/Seg3D/nnunetv2/training/models/mc_dropout_models/xmodules/0/5_7.json

wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 21 3d_fullres 0 -tr ourTrainer -model /home/andrewheschl/Documents/Seg3D/nnunetv2/training/models/mc_dropout_models/xmodules/0/3_11.json &
