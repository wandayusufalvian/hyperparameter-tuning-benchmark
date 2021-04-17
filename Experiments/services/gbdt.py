from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def xgboost_model():
    return XGBClassifier(verbosity = 0,use_label_encoder =False)

def lightgbm_model():
    return LGBMClassifier()

def catboost_model():
    return CatBoostClassifier()
