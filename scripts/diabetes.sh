#!/usr/bin/env bash

BASE_PATH='./..'

python $BASE_PATH/main.py --exp_name=Diabetes --num_clusters=3 --data_path=$BASE_PATH/Data/diabetes/ --pretrained_path=$BASE_PATH/Models/scvis_diabetes.pt --xydata
