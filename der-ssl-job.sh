#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3:0:0
#SBATCH --mem=32G
#SBATCH --account=def-bengioy
#SBATCH --mail-user=nikky.runghen.vezina@gmail.com
#SBATCH --mail-type=ALL

source ../clcomp21/bin/activate
./run_all.sh
