# hyperparameter-tuning-benchmark
Comparation of BOHB as hyperparameter optimization method with random search, grid search, and bayesian optimization to tune GBDT algorithm using 3 different dataset 

environment : 
- python 3.8.3
- anaconda 1.4.0
- Linux Ubuntu 18.04 

hyperparameter Optimization : 
- grid search
- random search
- bayesian optimization 
- BOHB (http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) 

GBDT Algorithms : 
- XGBoost 
- LightGBM
- CatBoost 

dataset : 
- [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Default Of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [Census Income Dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income)

experiments:
- experiment_1_xgboost_bank_default.py [done]
- experiment_2_xgboost_bank_gridsearch.py [done]
- experiment_3_xgboost_bank_randomsearch.py [done]
- experiment_4_xgboost_bank_bayessearch.py [done]
- experiment_5_1_xgboost_bank_bohb.py
  => memperlihatkan bahwa nilai seed tidak berpengaruh besar 
- experiment_5_2_xgboost_bank_bohb.py
  => variasi nilai n_iter. resources='n_samples'
- experiment_5_3_xgboost_bank_bohb.py
  => variasi nilai n_iter. resources='n_estimators'
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