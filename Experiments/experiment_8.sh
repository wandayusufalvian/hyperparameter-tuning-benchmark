#!/bin/bash

#SBATCH --job-name=experiment_8
#SBATCH --output=results/experiment_8_2.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi04

source /home/m448735/anaconda3/bin/activate
srun python experiment_8.py