#!/bin/bash 
#SBATCH -J TML
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output tml.output

srun ~/bin/exaflow3D incompressible_tml.ctr > log
