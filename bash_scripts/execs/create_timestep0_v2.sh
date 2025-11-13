#!/bin/bash
#SBATCH -A tur147
#SBATCH -J Init
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
set -e

#File locations
FPCSLPY="/ccs/home/abhi/source_code/envs/fpcsl"
INITIALIZATION_SCRIPT="/lustre/orion/tur147/scratch/abhi/incompressible-tml/TML_scripts/initialization_scripts"

#Load modules
module load miniforge3
conda deactivate
source activate $FPCSLPY

srun -n1 python3 $INITIALIZATION_SCRIPT/initialize_TML_v3.py --nx_g $nx_g --ny_g $ny_g --nz_g $nz_g \
                                                             --nxsd $nxsd --nysd $nysd --nzsd $nzsd \
                                                             --rho1 $rho1 --rho2 $rho2              \
                                                             --mu2  $mu2                            \
                                                             --U1   $U1   --Re   $Re   --We   $We                

mv *.png ./plots
mv *.out ./job_outputs
