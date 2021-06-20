#!/bin/bash

#SBATCH --job-name=node_2
#SBATCH --output=result_raw/exp_node_2.txt
#
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi07

source /home/m448735/anaconda3/bin/activate
srun python node_2.py