from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# xgboost_model=XGBClassifier(verbosity = 0,use_label_encoder =False)
# lightgbm_model=LGBMClassifier()
# catboost_model=CatBoostClassifier()


def xgboost_model():
    return XGBClassifier(verbosity = 0,use_label_encoder =False)

def lightgbm_model():
    return LGBMClassifier()

def catboost_model():
    return CatBoostClassifier()
