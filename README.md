# Explaining Low Dimensional Representations

> Plumb, Gregory, et al. "Explaining Groups of Points in Low-Dimensional Representations." International Conference on Machine Learning. PMLR, 2020.

This repository contains the implementation and reporduction of experiments presented in the above paper. The original paper proposes **Transitive Global Translations**, an algorithm to explain the differences between the _group of points_ in a low-dimensional space. In this repository, the claims and the results of the paper are studied and validated. Furthermore, we extend TGT with a scaling mechanism.

This README is organized as follows:

- Workflow
- Navigating the repository
- Contributors
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

## Contributors

Rajeev Verma, Paras Dahal, Jim Wagemans and Auke Elfrink, University of Amsterdam

## References

[1] _Jiarui Ding, Anne Condon, and Sohrab P. Shah._ Interpretable dimensionality reduction of single cell transcriptome data with deep generative models. bioRxiv, 2017.217

[2] _Dheeru Dua and Casey Graff._ UCI machine learning repository, 2017.N.B. The ‘Pima Indians Diabetes’ and ‘Boston Housing’ datasets are no longer available from this source.

[3] _Gregory Plumb, Jonathan Terhorst, Sriram Sankararaman, and Ameet Talwalkar._ Explaining groups of points in low-dimensional representations, 2020

[4] _Karthik Shekhar, Sylvain W Lapan, Irene E Whitney, Nicholas M Tran, Evan Macosko, Monika Kowalczyk, Xian222Adiconis, Joshua Z Levin, James Nemesh, Melissa Goldman, Steven Mccarroll, Constance L Cepko, Aviv Regev, and Joshua R Sanes._ Comprehensive classification of retinal bipolar neurons by single-cell transcriptomics. Cell,166:1308–1323.e30, 08 2016.
