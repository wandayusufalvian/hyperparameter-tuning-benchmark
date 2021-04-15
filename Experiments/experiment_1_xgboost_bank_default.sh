#!/bin/bash

#SBATCH --job-name=experiment_1_xgboost_bank_default
#SBATCH --output=experiment_1_xgboost_bank_default.txt
#
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

srun python experiment_1_xgboost_bank_default.py