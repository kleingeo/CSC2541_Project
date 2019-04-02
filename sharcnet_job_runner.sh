#!/bin/bash
#SBATCH --time=24:00
#SBATCH --account=def-amartel
#SBATCH --gres=gpu:lgpu:4  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --job-name=Brats_seg
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=24


python model_runner_sharcnet.py