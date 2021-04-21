#!/usr/bin/env bash
# 0.5 0.75 1 1.25 1.5 1.75 2
for x in 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7;
do
    python main_sweep.py --ssl_alpha=${x}
done