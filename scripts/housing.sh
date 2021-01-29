#!/usr/bin/env bash


cd ./../

python ./main.py --exp_name=Housing --num_cluster=6 --data_path=./ELDR/Housing/Data --pretrained_path=./Models/scvis_housing.pt --xydata >> housing
