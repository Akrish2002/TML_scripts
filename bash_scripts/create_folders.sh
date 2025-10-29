#!/bin/bash
set -e

TMLSCRIPTS_INITIALIZATION="/ccs/home/abhi/member-work/incompressible-tml/TML_scripts/initialization_scripts"
TMLSCRIPTS_BATCH="/ccs/home/abhi/member-work/incompressible-tml/TML_scripts/batch_scripts"

#Take in number of folders to create
echo -n "--Number of folders? "
read -r nfolders
echo -n "--Enter folder location: "
read -r PATH_TO_CREATED_FOLDERS

#Take arg for folder names
for i in $(seq 1 $nfolders)
do
    if [[ $i == 1 ]]; then
        echo -n "--Enter folder name: "
    else
        echo -n "--Next folder: "
    fi
    read -r fname 
    mkdir $fname

    #Making subfolders in each
    cd $fname
    mkdir run1
    cd run1
    mkdir "log"
    mkdir "plots"
    mkdir "postprocessed_data"

    #Copying the skeleton .ctr and .sl files from TML_scripts
    #cp $TMLSCRIPTS_INITIALIZATION/
    cp $TMLSCRIPTS_BATCH/batch.sl                   $PATH_TO_CREATED_FOLDERS
    cp $TMLSCRIPTS_BATCH/incompressible_tml.ctr     $PATH_TO_CREATED_FOLDERS

    #Leave folder
    cd ../../
done

