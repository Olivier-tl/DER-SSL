#!/usr/bin/env bash
# 0.5 0.75 1 1.25 1.5 1.75 2 2.5 3 3.5 4 4.5 5 5.5

for der_plus in 1 0;
do
    for x in 0.5 1.0 1.5 2.0;
    do
        if [[ $der_plus -eq 1 ]]
        then
            python main_sweep.py --beta=${x} --alpha=${x} --use_ssl=0 --use_drl=0 --model_name=\"resnet\"
        else
            python main_sweep.py --beta=0 --alpha=${x} --use_ssl=0 --use_drl=0 --model_name=\"resnet\"
        fi;
    done
done