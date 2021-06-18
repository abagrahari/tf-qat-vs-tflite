#!/bin/bash
rm summary.csv 2> /dev/null
rm summary_monkeypatch.csv 2> /dev/null
rm -r saved_weights/ 2> /dev/null

declare -a strings=("dense1" "dense2" "dense3" "dense4")

for j in "${strings[@]}"; do
    for i in `seq 0 3`; do
        python main.py --model $j --seed $i
        python main.py --model $j --seed $i --eval
    done
done

