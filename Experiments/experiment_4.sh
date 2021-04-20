#!/bin/bash

#SBATCH --job-name=experiment_4
#SBATCH --output=results/experiment_4_xgboost_bank_bayessearch.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi04

source /home/m448735/anaconda3/bin/activate
srun python experiment_4.py