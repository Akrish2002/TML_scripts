#!/bin/bash
#SBATCH -A tur147
#SBATCH -J n128
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1

export MPICH_GPU_SUPPORT_ENABLED=1
INPUT=incompressible_tml.ctr
EXEC=~/executables/bin/exaflow3D
INPUT=incompressible_tml.ctr

#Finding the latest file                                                    
latest_time_step=$(ls -1t time_step-* 2>/dev/null | head -n 1)              

#Finish this condition!!
#if [[ -z "${latest_time_step}]]; then
#    echo "No new time_step found. Exiting"                                  
#    exit 1                                                                  
#fi

new_time_step=$(echo "$latest_time_step" | grep -Po '\d+')              
#Updating the CTR file
sed -i "s/time_step = .*/time_step = $new_time_step/" "$INPUT"                  
                                                                        
#Running the simulation
srun --ntasks=8 --cpus-per-task=7 --gpus-per-task=1 $EXEC $INPUT > ./log/log_$(date +%Y-%m-%d_%H-%M-%S).out


