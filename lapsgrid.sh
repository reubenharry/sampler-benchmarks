#!/bin/bash

#SBATCH -A m4031_g
#SBATCH -N 1
#SBATCH --image=jrobnik/mcmc:1.0
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J LAPS
#SBATCH -t 03:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu


for IMODEL in {0..4}
do
  echo $IMODEL
  shifter python3 -m papers.LAPS.main $IMODEL 0
done


