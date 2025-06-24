#!/bin/sh
#SBATCH --nodes=1
#SBATCH --image=reubenharry/cosmo:1.0
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH -C cpu
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

shifter python3 results/ICG_2_1/main.py