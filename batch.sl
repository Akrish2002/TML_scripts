#!/bin/bash
#SBATCH -A tur147
#SBATCH -J sample-job
#SBATCH -o %x-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 4

export MPICH_GPU_SUPPORT_ENABLED=1

srun --ntasks=32 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest exaflow3D example.ctr
