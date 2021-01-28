#!/usr/bin/env bash

BASE_PATH='./..'

python $BASE_PATH/main.py --exp_name=Breast --num_clusters=3 --data_path=$BASE_PATH/Data/breast/ --pretrained_path=$BASE_PATH/Models/scvis_breast.pt --xydata
