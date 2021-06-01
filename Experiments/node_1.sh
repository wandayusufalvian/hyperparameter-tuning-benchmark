#!/bin/bash

#SBATCH --job-name=node_1
#SBATCH --output=temp_result/exp_17_22_27.txt
#
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

source /home/m448735/anaconda3/bin/activate
srun python node_1.py