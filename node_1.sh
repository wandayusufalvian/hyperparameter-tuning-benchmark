#!/bin/bash

#SBATCH --job-name=node_1
#SBATCH --output=result_raw/exp_18_19.txt
#
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi04

source /home/m448735/anaconda3/bin/activate
srun python node_1.py