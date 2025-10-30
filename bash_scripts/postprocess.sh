#!/bin/bash
set -e

#Breaking down postprocessing

#- srun -n y-stack_cores
#- Changing file name of .csv                                                   --> For integrand data it greps from ctr file
#- Redirecting postprocessed data to processed data folder
#- asdf

echo "--Submitting batch job to postprocess data..."

#Script to compute growth rate
echo "--Computing growth rate.."
sbatch compute_growthrate.sh

#Script to compute time
echo "--Computing normalized time.."
compute_normalizedtime.sh



