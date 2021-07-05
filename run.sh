#!/bin/bash
rm summary.csv 2> /dev/null
rm summary_monkeypatch.csv 2> /dev/null
rm -r saved_weights/ 2> /dev/null
rm -r saved_models/ 2> /dev/null

for i in `seq 0 3`; do
    python main.py --seed $i
    python main.py --seed $i --eval
done

