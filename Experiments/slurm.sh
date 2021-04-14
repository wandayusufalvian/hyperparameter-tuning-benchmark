#!/bin/bash
####sample_job.sh####
#SBATCH --job-name=sample_job
#SBATCH --output=results.txt
#
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi04

srun hello.py