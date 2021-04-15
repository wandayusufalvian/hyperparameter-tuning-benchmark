#!/bin/bash

#SBATCH --job-name=experiment_1
#SBATCH --output=experiment_1.txt
#
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06

srun python hello.py