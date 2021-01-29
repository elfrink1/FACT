#!/usr/bin/env bash


cd ./../

python ./main.py --exp_name=Breast_Scaling --use_scaling --num_clusters=3 --data_path=./Data/breast/ --pretrained_path=./Models/scvis_breast.pt --xydata >> ./loggers/breastCancerScaling
