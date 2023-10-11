#!/bin/bash
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --mail-user=andrew.heschl@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --gres=gpu:2

#Environment setup
source ~/software/init-conda
conda activate torch_env

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 29 2d 0 -tr ourTrainer -model /home/andrew.heschl/Documents/599_Architecture_Project/nnunetv2/training/models/599_models/poly_v2/fact_poly.json &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 29 2d 0 -tr ourTrainer -model /home/andrew.heschl/Documents/599_Architecture_Project/nnunetv2/training/models/599_models/poly_v2/sum_poly.json
