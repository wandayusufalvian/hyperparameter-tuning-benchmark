import numpy as np 
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real,Integer
from scipy.stats import uniform
from scipy.stats import loguniform
    
def hyper_xgboost_gs():
    cs={
        'eta': [0.01,0.5], 
        'subsample': [0.1,0.8],
        'max_depth': [2,8,14,20,26],
        'gamma':[0.001,0.4],
        'min_child_weight':[2,5,7,11,15]
    }
    return cs

def hyper_xgboost_rs():
    cs={
        'eta': loguniform(1e-5,1),
        'subsample': uniform(0.1,0.9),
        'max_depth':list(range(1,99)),
        'gamma': uniform(0.001,1.999),
        'min_child_weight': uniform(1,69)
    }
    return cs 

def hyper_xgboost_bo():
    cs={
        'eta': Real(1e-5,1,'log-uniform'), 
        'subsample': Real(0.1,1,'uniform'),
        'max_depth': Integer(1,100,'uniform'),
        'gamma': Real(0.001,2,'uniform'),
        'min_child_weight': Real(1,70,'uniform')
    }
    return cs 

def hyper_xgboost_bohb():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('eta',lower=1e-5,upper=1,log=True))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('subsample',lower=0.1,upper=1,log=False))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter
                          ('max_depth',lower=1,upper=100,log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('gamma',lower=0.001,upper=2,log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('min_child_weight',lower=1,upper=70,log=False))
    return cs

def hyper_lightgbm_gs(): 
    cs={
        'max_depth': [10,20,30,40,50],
        'min_data_in_leaf':[10,30,50,70,90],
        'num_leaves': [10,40],
        'learning_rate': [0.01,0.5], 
        'bagging_fraction':[0.1,0.9]
    }
    return cs 

def hyper_lightgbm_rs():
    cs={
        'max_depth': list(range(1,101)),
        'min_data_in_leaf':list(range(1,101)),
        'num_leaves': list(range(2,101)),
        'learning_rate': loguniform(1e-5,1),
        'bagging_fraction':uniform(0.1,0.90) 
    }
    return cs 

def hyper_lightgbm_bo():
    cs={
        'max_depth': Integer(1,100,'uniform'),
        'min_data_in_leaf': Integer(1,100,'uniform'),
        'num_leaves': Integer(2,100,'uniform'),
        'learning_rate': Real(1e-5,1,'log-uniform'),
        'bagging_fraction':Real(0.1,1,'uniform')    
    }
    return cs 

def hyper_lightgbm_gs_2(): 
    cs={
        'max_depth': [5,30,50,70,90],
        'scale_pos_weight': [3,5,7,9,14],
        'num_leaves': [30,400],
        'bagging_fraction': [0.45,0.55],
        'colsample_bytree': [0.45,0.55]
    }
    return cs 

def hyper_lightgbm_rs_2():
    cs={
        'max_depth': list(range(2,101)),
        'scale_pos_weight': uniform(1,19),
        'num_leaves': list(range(10,1001)),
        'bagging_fraction': uniform(0.4,0.6),
        'colsample_bytree':uniform(0.4,0.6)    
    }
    return cs 

def hyper_lightgbm_bo_2():
    cs={
        'max_depth': Integer(2,100,'uniform'),
        'scale_pos_weight': Real(1,20,'uniform'),
        'num_leaves': Integer(10,1000,'uniform'),
        'bagging_fraction': Real(0.4,1,'uniform'),
        'colsample_bytree':Real(0.4,1,'uniform')    
    }
    return cs 

def hyper_lightgbm_bohb():
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter
                          ('max_depth',lower=1,upper=100,log=False))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter
                          ('min_data_in_leaf',lower=1,upper=100,log=False))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter
                          ('num_leaves',lower=1,upper=100,log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('learning_rate',lower=1e-5,upper=1,log=True))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('bagging_fraction',lower=0.1,upper=1,log=False))
    return cs 

def hyper_catboost_gs():
    cs={
        'max_depth': [2,5,8,11,14],
        'learning_rate': [0.01,0.1],
        'l2_leaf_reg':[1,6,11,16,21],
        'bagging_temperature':[0.5,20],
        'random_strength':[0.5,20]
    }
    return cs 

def hyper_catboost_rs():
    cs={
        'max_depth': list(range(1,17)),
        'learning_rate': loguniform(0.001,1),
        'l2_leaf_reg':uniform(1,49),
        'bagging_temperature':uniform(1,49),
        'random_strength':uniform(1,49)
    }
    return cs 

def hyper_catboost_bo():
    cs={
        'max_depth': Integer(1,16,'uniform'),
        'learning_rate': Real(0.001,1,'log-uniform'),
        'l2_leaf_reg': Real(1,50,'uniform'),
        'bagging_temperature':Real(1,50,'uniform'),
        'random_strength':Real(1,50,'uniform')
    }
    return cs 

def hyper_catboost_bohb():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter
                          ('max_depth',lower=1,upper=16,log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('learning_rate',lower=0.001,upper=1,log=True))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('l2_leaf_reg',lower=1,upper=50,log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('bagging_temperature',lower=1,upper=50,log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter
                          ('random_strength',lower=1,upper=50,log=False))
    return cs