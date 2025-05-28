#!/bin/bash
#SBATCH -JTML512
#SBATCH -A gts-ssuresh313-startup
#SBATCH -t5:00:00
#SBATCH -n256 
#SBATCH -otml.output

srun ~/abhijeet/bin/exaflow3D incompressible_tml.ctr > log
