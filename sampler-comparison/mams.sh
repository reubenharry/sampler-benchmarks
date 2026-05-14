#!/bin/bash

#SBATCH -A m4031
#SBATCH -N 1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J MAMS
#SBATCH -t 03:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


shifter python3 -m MAMS_PAPER_2025.convergence_curves
