#!/bin/bash
#
#SBATCH --partition=sc-quick 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 3
#SBATCH --mem 8GB

#SBATCH --x11

python $1