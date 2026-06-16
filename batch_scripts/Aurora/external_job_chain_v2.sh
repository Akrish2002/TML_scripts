#!/bin/bash
set -euo pipefail

SLEEP=2280  #38 mins
POLL_SLEEP=30

LOGDRIVER="driver_14-07-2026.txt"

SCRIPT="batch_v3.qsub"
MAX_CHAIN=5 


for i in $(seq 1 "$MAX_CHAIN"); do
    jid_full=$(qsub -v CHAIN_INDEX="$i",MAX_CHAIN="$MAX_CHAIN" "$SCRIPT")
    jid=${jid_full%%.*}
    echo -e "\n###########################"  | tee -a $LOGDRIVER
    echo -e "Submitted Job ${i}: ${jid}" | tee -a $LOGDRIVER

    outfile="n1024_We05_mu_ratio1.0.o${jid}"
    while [[ ! -f "$outfile" ]]; do
        sleep 90
    done
    echo "Output file created: $outfile" | tee -a $LOGDRIVER

    #Wait while job reaches its end
    sleep $SLEEP

    #Wait while job exists in PBS
    while qstat "$jid" >/dev/null 2>&1; do
        sleep $POLL_SLEEP
    done

    #Check latest dump sizes
    latest_dir=$(ls -1dt time_step-* 2>/dev/null | head -n 1 || true)

    echo "Job ${i} finished: ${jid}"       | tee -a $LOGDRIVER
    echo "${latest_dir} has been created!" | tee -a $LOGDRIVER

    echo -e "----------------------------------"  | tee -a $LOGDRIVER

    echo -e "Checking dump sizes in $latest_dir" | tee -a "$LOGDRIVER"
    bad_dump=0
    for f in "$latest_dir"/*.raw; do
        size=$(du -m "$f" | awk '{print $1}')
        if [[ "$size" != "264" ]]; then
            echo "ERROR: $f is $size, expected 264M" | tee -a "$LOGDRIVER"
            bad_dump=1
        fi
    done

    for f in "$latest_dir"/*.vtr; do
        size=$(du -m "$f" | awk '{print $1}')
        if [[ "$size" != "241" ]]; then
            echo "ERROR: $f is $size, expected 241M" | tee -a "$LOGDRIVER"
            bad_dump=1
        fi
    done

    if [[ "$bad_dump" -ne 0 ]]; then
        echo "Dump check failed. Stopping chain." | tee -a "$LOGDRIVER"
        echo -e "###########################"  | tee -a $LOGDRIVER
        break
    fi

    echo "Dump check passed: all .raw files are 264M and all .vtr files are 241M. Continuing." | tee -a "$LOGDRIVER"
    echo -e "###########################"  | tee -a $LOGDRIVER
done

