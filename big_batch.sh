#!/bin/bash
#
#SBATCH --partition=sc-quick 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --mem 25GB

#SBATCH --x11

python $1