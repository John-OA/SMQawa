#!/bin/bash
set -uo pipefail
TAGGERS=("deeptau" "pnet" "upart")
mkdir -p run_logs
for tagger in "${TAGGERS[@]}"; do mkdir -p "output/${tagger}"; done

mapfile -t DATASETS < small_test_datasets.txt

for dsname in "${DATASETS[@]}"; do
    [ -z "$dsname" ] && continue
    primary=$(echo "$dsname" | cut -d'/' -f2)
    alias=$(echo "$primary" | sed -E 's/_TuneCP5.*//' | tr -c 'A-Za-z0-9' '_')

    echo "=== $alias -> $dsname ==="
    infile=""
    for candidate in $(dasgoclient -query="file dataset=$dsname" | head -n10); do
        if xrdfs root://eoscms.cern.ch/ stat "$candidate" > /dev/null 2>&1; then
            infile="/eos/cms${candidate}"
            break
        fi
    done
    if [ -z "$infile" ]; then
        echo "  WARNING: no accessible file found in first 10, skipping"
        continue
    fi

    for tagger in "${TAGGERS[@]}"; do
        outpath="output/${tagger}/histogram_${alias}.pkl.gz"
        if [ -f "$outpath" ]; then echo "  $outpath exists, skipping"; continue; fi
        python3 brewer-remote-inclusive.py \
            --isMC=1 --era=2024 --infile="$infile" \
            --executor=FuturesExecutor --maxChunks=10 --htau_tagger="$tagger" \
            > "run_logs/${alias}_${tagger}.log" 2>&1
        wait
        sleep 5
        if [ -f histogram_1.pkl.gz ]; then
            mv histogram_1.pkl.gz "$outpath"
            echo "  -> $outpath"
        else
            echo "  FAILED (run_logs/${alias}_${tagger}.log)"
        fi
    done
done
echo "DONE"
