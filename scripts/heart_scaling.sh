#!/usr/bin/env bash


BASE_PATH='./..'

python $BASE_PATH/main.py --exp_name=Heart_scaling --use_scaling --num_cluster=4 --data_path=$BASE_PATH/ELDR/Heart/Data --pretrained_path=$BASE_PATH/Models/scvis_heart.pt --xydata
