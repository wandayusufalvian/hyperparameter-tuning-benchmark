# hyperparameter-tuning-benchmark

## Purpose of research 

Optimize hyperparameter of 3 most popular GBDT Algorithms : XGBoost, LightGBM, and CatBoost using BOHB, state of the art Hyperparameter Optimization (HPO) algorithm. For comparison to BOHB performance, i use 3 others HPO algorithm : Grid Search, Random Search, and Bayesian Optimization.

## What is BOHB? 

BOHB is state of the art Hyperparameter Optimization algorithm that was  developed by Falkner, et.al (2018) : http://proceedings.mlr.press/v80/falkner18a.html

## Dataset : 

3 kind of dataset : imbalance binary class, multiclass, and numerical dataset. 

- dataset 1 (imbalance binary class dataset) : https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- dataset 2 (multiclass dataset) : https://www.kaggle.com/abisheksudarshan/customer-segmentation
- dataset 3 (numerical dataset) : https://www.kaggle.com/anmolkumar/house-price-prediction-challenge

## Environment : 
- python 3.8.3
- anaconda 1.4.0
- Linux Ubuntu 18.04 

## How to run experiments? 
I run these experiments in HPC (High Performance Computer). I make 2 .sh files to run experiments in HPC :
- node_1.sh
- node_2.sh 

node_1.sh will execute node_1.py and node_2.sh will execute node_2.py.
change this code in node_1.sh or node_2.sh to save the file
```
#SBATCH --output=result_raw/your-file-name.txt 
```

The result will be save in ~/result_raw/your-file-name.txt
