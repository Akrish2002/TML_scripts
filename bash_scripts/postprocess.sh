#!/bin/bash
set -e

#Breaking down postprocessing

#- srun -n y-stack_cores
#- Changing file name of .csv                                                   --> For integrand data it greps from ctr file
#- Redirecting postprocessed data to processed data folder
#- asdf

echo "--Functions:
        [1] Growth Rate
        [2] Compute times from log file(s)
        [3] asdf
     "
read -r arg

if [[ $arg == "1" ]]; then
    #Script to compute growth rate
    echo "--Submitting batch job to compute growth rate.."
    sbatch compute_growthrate.sh

elif [[ $arg == "2" ]]; then
    #Script to compute time
    echo "--Computing normalized time.."
    compute_normalizedtime.sh

else 
    echo"--Nothing entered!"

fi


