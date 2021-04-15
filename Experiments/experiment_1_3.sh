#!/bin/bash

#SBATCH --job-name=experiment_1_3
#SBATCH --output=results/experiment_1_4.txt
#
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python experiment_1_2.py