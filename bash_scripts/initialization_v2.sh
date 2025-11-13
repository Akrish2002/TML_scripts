#!/bin/bash
set -euo pipefail


echo "--Enter the following parameters: "

#1. Discretization
echo "--Control volumes: "
read -r nx_g
read -r ny_g
read -r nz_g

echo "--nxsd, nysd and nzsd: "
read -r nxsd 
read -r nysd
read -r nzsd

#2. Parameters
echo "--ρ1 and ρ2: "
read -r rho1 
read -r rho2

echo "--μ1 and μ2: "
read -r mu1 
read -r mu2

echo "--Re and We: "
read -r Re 
read -r We

#Making the vars as env variables so it is visible to create_timestep0.sh
nx_g="$nx_g"
ny_g="$ny_g"
nz_g="$nz_g"

nxsd="$nxsd"
nysd="$nysd"
nzsd="$nzsd"

rho1="$rho1"
rho2="$rho2"

mu1="$mu1"
mu2="$mu2"

Re="$Re"
We="$We"

export nx_g ny_g nz_g
export nxsd nysd nzsd
export rho1 rho2
export mu1 mu2
export Re We

sbatch create_timestep0_v2.sh

