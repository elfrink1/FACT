# Explaining Low Dimensional Representations

> Plumb, Gregory, et al. "Explaining Groups of Points in Low-Dimensional Representations." International Conference on Machine Learning. PMLR, 2020.

This repository contains the implementation and reproduction of experiments presented in the above paper. The original paper proposes **Transitive Global Translations**, an algorithm to explain the differences between the _group of points_ in a low-dimensional space. In this repository, the claims and the results of the paper are studied and validated. Furthermore, we extend TGT with a scaling mechanism.

This README is organized as follows:

- Workflow
- Navigating the repository
- Running the experiments
- Contributors
- Acknowledgements
- References

## Workflow

It is recommended to adhere to the following workflow for using this repository. Please use the scripts in `scripts` directory to run the experiments.
![Workflow](https://github.com/elfrink1/FACT/blob/main/imgs/workflow.png?raw=true)

## Navigating the repository

```
.
+-- Data (Data files used in the experiments)
+-- ELDR-reproduction (Reproduction of experiments from author's code)
+-- Environment (YAML environment files for Conda)
+-- Models (Pretrained models from our experiments)
+-- configs (JSON config files for TGT, AE and VAE models)
+-- eldr (Our Pytorch implementation)
+-- experiments (IPython notebooks of our extension experiments)
+-- scripts (Bash scripts for running experiments with different datasets)
+-- main.py (Main entry file for running experiments with CLI)
+-- trainr.py (Script to train low dimensional representations)
```
## Running the experiments
We provide the trained models and explanations. To run, navigate to the corresponding experiment in the ./experiments dir, and follow the .ipynb notebook.

We provide the following environments file:
| file_name | env_name|
|-----------|---------|
|lisa_tensorflow_env.yml | tffact|
|pytorch_env.yml| factai|

To install the environment:
```
conda env create -f $filename.yml
conda activate $env_name
```

As described in the workflow above, `trainr.py` trains the low-dimensional representation learning function.
The usage is described as below:

```
usage: trainr.py [-h] [--model_type {vae,autoencoder}]
                 [--pretrained_path PRETRAINED_PATH] [--model_dir MODEL_DIR]
                 [--data_path DATA_PATH] [--train] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --model_type {vae,autoencoder}
                        Type of model for Learning low dimensional
                        representations (default: vae)
  --pretrained_path PRETRAINED_PATH
                        Path to the trained model (default: ./Models/vae.pt)
  --model_dir MODEL_DIR
                        Path to save the trained model (default: ./Models)
  --data_path DATA_PATH
                        Path of the data to use (default: ./ELDR/Housing/Data)
  --train               Do you want to train? (default: False)
  --dataset DATASET     Dataset on which you are training or equivalently
                        exp_name. Trained model will be saved with this name.
                        (default: random)
```
The script `main.py` trains the explanations between the groups. It's usage description is as follows:

```
usage: main.py [-h] [--model_type {vae,autoencoder}]
               [--pretrained_path PRETRAINED_PATH] [--data_path DATA_PATH]
               [--num_clusters NUM_CLUSTERS] [--xydata] [--exp_name EXP_NAME]
               [--use_scaling]

optional arguments:
  -h, --help            show this help message and exit
  --model_type {vae,autoencoder}
                        Type of model for Learning low dimensional
                        representations (default: vae)
  --pretrained_path PRETRAINED_PATH
                        Path to the trained model (default: ./Models/vae.pt)
  --data_path DATA_PATH
                        Path of the data to use (default: ./ELDR/Housing/Data)
  --num_clusters NUM_CLUSTERS
                        Number of Clusters (default: 6)
  --xydata              Labels and data stored seperately (default: False)
  --exp_name EXP_NAME   Name of the experiment. Everything will be saved at
                        ./experiments/$exp_name$ (default: Housing)
  --use_scaling         Use extended explanations with exponential scaling
                        (default: False)
```

## Acknowledgements
We appreciate the original authors for making the code public. We would like to thank the lead author Gregory Plumb for promptly replying to the emails for suggestions and help. Finally, a great thanks to Christina Winkler for the feedback throughout the project.


## Contributors

Rajeev Verma, Paras Dahal, Jim Wagemans and Auke Elfrink, University of Amsterdam

## References

[1] _Jiarui Ding, Anne Condon, and Sohrab P. Shah._ Interpretable dimensionality reduction of single cell transcriptome data with deep generative models. bioRxiv, 2017.217

[2] _Dheeru Dua and Casey Graff._ UCI machine learning repository, 2017.N.B. The ‘Pima Indians Diabetes’ and ‘Boston Housing’ datasets are no longer available from this source.

[3] _Gregory Plumb, Jonathan Terhorst, Sriram Sankararaman, and Ameet Talwalkar._ Explaining groups of points in low-dimensional representations, 2020

[4] _Karthik Shekhar, Sylvain W Lapan, Irene E Whitney, Nicholas M Tran, Evan Macosko, Monika Kowalczyk, Xian222Adiconis, Joshua Z Levin, James Nemesh, Melissa Goldman, Steven Mccarroll, Constance L Cepko, Aviv Regev, and Joshua R Sanes._ Comprehensive classification of retinal bipolar neurons by single-cell transcriptomics. Cell,166:1308–1323.e30, 08 2016.
