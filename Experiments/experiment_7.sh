#!/bin/bash

#SBATCH --job-name=experiment_7
#SBATCH --output=results/experiment_7_2.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi04

source /home/m448735/anaconda3/bin/activate
srun python experiment_7.py