#!/bin/bash

#SBATCH --job-name=experiment_2
#SBATCH --output=results/experiment_2_xgboost_bank_gridsearch.txt
#
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python experiment_2.py