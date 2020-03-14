#!/bin/bash

# # Activate conda environment
# source activate $ME

# Change directory
cd src

# Define constants
ARCHS=("siamdiff" "siamconc" "EF")
DATASETS=("AC_Szada" "AC_Tiszadob" "OSCD")

# LOOP
for arch in ${ARCHS[@]}
do
    for dataset in ${DATASETS[@]}
    do
        python train.py train --exp-config ../config_${arch}_${dataset}.yaml
    done
done