#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=6:0:0
#SBATCH --mem=64000
#SBATCH --account=def-bengioy
#SBATCH --mail-user=nikky.runghen.vezina@gmail.com
#SBATCH --mail-type=ALL

source ../clcomp21/bin/activate
./run_all.sh
python py_run_all.py
