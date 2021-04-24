from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def xgboost_model():
    return XGBClassifier(verbosity = 0,use_label_encoder =False)

def lightgbm_model():
    return LGBMClassifier()

def catboost_model():
    return CatBoostClassifier(logging_level='Silent')

def catboost_model_2():
    return CatBoostClassifier(logging_level='Silent',iterations=500,boosting_type='Plain',rsm=0.1,max_ctr_complexity=2,)