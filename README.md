# hyperparameter-tuning-benchmark
Comparation of BOHB as hyperparameter optimization method with random search, grid search, and bayesian optimization to tune GBDT algorithm using 3 different dataset 

## Environment : 
- python 3.8.3
- anaconda 1.4.0
- Linux Ubuntu 18.04 

## hyperparameter Optimization : 
- grid search
- random search
- bayesian optimization 
- BOHB (http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) 

## GBDT Algorithms : 
- XGBoost 
- LightGBM
- CatBoost 

## dataset : 
- [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Default Of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [Census Income Dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income)

## Experiments 
* change directory to `cd/Experiments`
* there are 2 .py file = node1.py and node2.py. I use two files to run simultaneously in 2 different HPC node
* you can choose to run locally or in HPC. use .sh file to run on HPC 
* file name of the file that contain results should be re-written in .sh if you want to run on HPC 
* results will be stored in Experiments/results
