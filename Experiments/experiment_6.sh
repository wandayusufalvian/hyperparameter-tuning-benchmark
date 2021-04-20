#!/bin/bash

#SBATCH --job-name=experiment_6
#SBATCH --output=results/experiment_6_lightgbm_bank_default.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python experiment_6.py