#!/usr/bin/env bash

cd ./../

python ./main.py --exp_name=Diabetes --num_clusters=3 --data_path=./Data/diabetes/ --pretrained_path=./Models/scvis_diabetes.pt --xydata
