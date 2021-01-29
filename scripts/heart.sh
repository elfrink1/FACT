#!/usr/bin/env bash


cd ./../

python main.py --exp_name=Heart --num_cluster=4 --data_path=./ELDR/Heart/Data --pretrained_path=./Models/scvis_heart.pt --xydata >> ./loggers/heart
