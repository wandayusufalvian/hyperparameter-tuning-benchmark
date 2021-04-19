#!/bin/bash

#SBATCH --job-name=experiment_5
#SBATCH --output=results/experiment_5_xgboost_bank_bohb.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python experiment_5_xgboost_bank_bohb.py