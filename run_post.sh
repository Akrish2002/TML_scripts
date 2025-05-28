#!/bin/bash 
#SBATCH -J postproc
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=192G
#SBATCH --output post.output


srun python postprocess.py > log_post
