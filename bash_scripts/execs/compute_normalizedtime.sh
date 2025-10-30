#!/bin/bash
set -euo pipefail

#File locations
FPCSLPY="/ccs/home/abhi/source_code/envs/fpcsl"
PYTHONSCRIPT="/ccs/home/abhi/member-work/incompressible-tml/TML_scripts/postprocessing_scripts"

#Load modules
module load miniforge3
conda deactivate
source activate $FPCSLPY

#Running the script on the login node, its fine!
python3 $PYTHONSCRIPT/grep_logTime.py

mv *.csv ./postprocessed_data
