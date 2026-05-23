#!/bin/bash
set -euo pipefail

SLEEP=2310  #38.5 mins
POLL_SLEEP=30

LOGDRIVER="driver_22-06-2026.txt"

SCRIPT="batch_v2.qsub"
MAX_CHAIN=20


for i in $(seq 1 "$MAX_CHAIN"); do
    jid_full=$(qsub -v CHAIN_INDEX="$i",MAX_CHAIN="$MAX_CHAIN" "$SCRIPT")
    jid=${jid_full%%.*}
    echo -e "\nSubmitted Job ${i}: ${jid}" | tee -a $LOGDRIVER

    outfile="n1024_v2.o${jid}"
    while [[ ! -f "$outfile" ]]; do
        sleep 90
    done
    echo "Output file created: $outfile"

    #Wait while job reaches its end
    sleep $SLEEP

    #Wait while job exists in PBS
    while qstat "$jid" >/dev/null 2>&1; do
        sleep $POLL_SLEEP
    done

    echo "Job ${i} finished: ${jid}" | tee -a $LOGDRIVER
done
