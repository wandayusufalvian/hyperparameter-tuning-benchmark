#!/bin/bash

#SBATCH --job-name=experiment_1
#SBATCH --output=experiment_1.txt
#
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source ~/anaconda3/bin/activate
srun python experiment_1_xgboost_bank_default.py