#!/usr/bin/env bash


BASE_PATH='./..'

python $BASE_PATH/main.py --exp_name=Housing_scaling --use_scaling --num_cluster=6 --data_path=$BASE_PATH/ELDR/Housing/Data
