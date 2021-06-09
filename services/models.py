from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def xgboost_model():
    return XGBClassifier(verbosity = 0,use_label_encoder =False)

def lightgbm_model(verbose=-1):
    return LGBMClassifier()

def catboost_model():
    # udah gk dipake karena run nya terlalu lama 
    return CatBoostClassifier(logging_level='Silent')

def catboost_model_2():
    return CatBoostClassifier(logging_level='Silent',iterations=500,boosting_type='Plain',rsm=0.1,max_ctr_complexity=2,allow_writing_files=False)
