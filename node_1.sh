#!/bin/bash

#SBATCH --job-name=node_1
#SBATCH --output=result_raw/exp_17_22.txt
#
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi03

source /home/m448735/anaconda3/bin/activate
srun python node_1.py