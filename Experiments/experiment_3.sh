#!/bin/bash

#SBATCH --job-name=experiment_3
#SBATCH --output=results/experiment_3_xgboost_bank_randomsearch.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python experiment_3_xgboost_bank_randomsearch.py