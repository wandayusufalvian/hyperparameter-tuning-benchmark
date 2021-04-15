# hyperparameter-tuning-benchmark
Comparation of BOHB as hyperparameter optimization method with random search, grid search, and bayesian optimization to tune GBDT algorithm using 3 different dataset 

Requirements : 
- python 3.7 
- scikit-learn  0.24 
- numpy 1.16.2 
- pandas 0.24.2 
- configspace 0.4.17
- scikit-optimize 0.8.1
- xgboost 1.3.1
- lightgbm 3.1.1
- catboost 0.24.4 

Hyperparameter Optimization : 
- grid search
- random search
- bayesian optimization 
- BOHB (http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) 

GBDT Algorithms : 
- XGBoost 
- LightGBM
- CatBoost 

Dataset : 
- [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Default Of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [Census Income Dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income)

Experiments:
- experiment_1_xgboost_bank_default.py 
- experiment_2_xgboost_bank_gridsearch.py
- experiment_3_xgboost_bank_randomsearch.py
- experiment_4_xgboost_bank_bayessearch.py
- experiment_5_xgboost_bank_bohb.py
- experiment_6_lightgbm_bank_default.py 
- experiment_7_lightgbm_bank_gridsearch.py
- experiment_8_lightgbm_bank_randomsearch.py
- experiment_9_lightgbm_bank_bayessearch.py
- experiment_10_lightgbm_bank_bohb.py
- experiment_11_catboost_bank_default.py 
- experiment_12_catboost_bank_gridsearch.py
- experiment_13_catboost_bank_randomsearch.py
- experiment_14_catboost_bank_bayessearch.py
- experiment_15_catboost_bank_bohb.py