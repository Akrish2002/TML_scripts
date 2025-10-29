#!/bin/bash
set -e

#Folder name
echo -n "-- Folder name to delete? "
read -r fname

echo -n "-- Do you still wish to delete? "
read -r ans
if [[ $ans == "Yes!" ]]; then
    rm -rf $fname
fi
#if [[ -n "$ans" ]]; then
#    echo
