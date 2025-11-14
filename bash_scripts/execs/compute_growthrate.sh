#!/bin/bash
#SBATCH -A tur147
#SBATCH -J postprocessing
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
set -e

#File locations
FPCSLPY="/ccs/home/abhi/source_code/envs/fpcsl"
PYTHONSCRIPT="/ccs/home/abhi/member-work/incompressible-tml/TML_scripts/postprocessing_scripts"

#Load modules
module load miniforge3
conda deactivate
source activate $FPCSLPY

#Grepping ny_g
nysd=$(python3 $PYTHONSCRIPT/postprocess_parallelcore_v1.py grepnysd)
srun -n$nysd python3 $PYTHONSCRIPT/postprocess_parallelcore_v1.py 

#Copying the data to plot/data for plotting
cp *.csv $PATH_TO_DATA

mv *.csv ./postprocessed_data 
mv *.out ./job_outputs 

mv *.out ./job_outputs 


