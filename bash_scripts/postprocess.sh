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
        [3] Plotting growth rates
        [4] asdf
     "
read -r arg

echo "--Enter location to dump data: "                                          
read -r PATH_TO_DATA
#Making the path to data an evn variable
PATH_TO_DATA="$PATH_TO_DATA"
export PATH_TO_DATA

if [[ $arg == "1" ]]; then
    #Script to compute growth rate
    echo "--Submitting batch job to compute growth rate.."
    sbatch compute_growthrate.sh

elif [[ $arg == "2" ]]; then
    #Script to compute time
    echo "--Computing normalized time.."
    compute_normalizedtime.sh

elif [[ $arg == "3" ]]; then
    #Script to plot growth rates
    echo "--Enter folder location to dump plots: "
    read -r PATH_TO_PLOTS
    PATH_TO_PLOTS="$PATH_TO_PLOTS"
    export PATH_TO_PLOTS
    echo "--Plotting growth rates.."
    plot_growthrate.sh 

else 
    echo"--Nothing entered!"

fi


