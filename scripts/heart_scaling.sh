#!/usr/bin/env bash


cd ./../

python ./main.py --exp_name=Heart_scaling --use_scaling --num_cluster=4 --data_path=./ELDR/Heart/Data --pretrained_path=./Models/scvis_heart.pt --xydata >> heartScaling
