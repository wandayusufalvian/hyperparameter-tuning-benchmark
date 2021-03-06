from xgboost import XGBClassifier,XGBRegressor 
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor

# classifier 

def xgboost_model():
    return XGBClassifier(verbosity = 0,use_label_encoder =False)

def xgboost_model_reg():
    return XGBRegressor(verbosity = 0,use_label_encoder =False)

def lightgbm_model():
    return LGBMClassifier(verbose=-1)

def lightgbm_model_reg():
    return LGBMRegressor(verbose=-1)


def catboost_model():
    # udah gk dipake karena run nya terlalu lama 
    return CatBoostClassifier(logging_level='Silent')

def catboost_model_2():
    return CatBoostClassifier(logging_level='Silent',iterations=500,boosting_type='Plain',rsm=0.1,max_ctr_complexity=2,allow_writing_files=False)


def catboost_model_2_reg():
    return CatBoostRegressor(logging_level='Silent',iterations=500,boosting_type='Plain',rsm=0.1,max_ctr_complexity=2,allow_writing_files=False)