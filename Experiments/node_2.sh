#!/bin/bash

#SBATCH --job-name=node_2
#SBATCH --output=results/experiment_12_catboost_bank_gridsearch.txt
#
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi04

source /home/m448735/anaconda3/bin/activate
srun python node_2.py