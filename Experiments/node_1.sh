#!/bin/bash

#SBATCH --job-name=node_1
#SBATCH --output=results/experiment_11_catboost_bank_default_2.txt
#
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python node_1.py