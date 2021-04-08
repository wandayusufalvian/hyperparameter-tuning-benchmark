from xgboost import XGBClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import time
from services.data import *
from services.hpo import *
from services.hyperparameter import * 

model_x=XGBClassifier(verbosity = 0,use_label_encoder =False)

X_bank,y_bank=baca_data_bank()
X_credit,y_credit=baca_data_credit()
X_census,y_census=baca_data_credit()

def xgboost_bank_default():
    start_time = time.time()

    skf=StratifiedKFold(n_splits=5)
    results=cross_val_score(model_x,X_bank,y_bank,cv=skf,scoring="roc_auc")

    end_time = time.time()

    list_hasil=[str(round(results.mean(),4)),str(round(results.std(),4)),str(end_time-start_time)]
    simpan_hasil("xgboost_bank_default",list_hasil)
 

def xgboost_bank_gs():
    start_time = time.time()

    auc,std,prmtr,best_index=grid_search(X_bank,y_bank,model_x,hyper_dummy(),"roc_auc")
    best_auc=auc[0][best_index]
    best_std=std[0][best_index]
    best_param=prmtr[0][best_index]

    end_time = time.time()

    temp=[auc,std,prmtr,best_index,best_auc,best_std,best_param,end_time-start_time]

    list_hasil=[str(i) for i in temp]
    simpan_hasil("xgboost_bank_gs",list_hasil)


def xgboost_bank_rs():
    
    return 0 

def xgboost_bank_bo():
    return 0 

def xgboost_bank_bohb():
    return 0

def xgboost_credit_default():
    return 0 

def xgboost_credit_gs():
    return 0 

def xgboost_credit_rs():
    return 0 

def xgboost_credit_bo():
    return 0 

def xgboost_credit_bohb():
    return 0

def xgboost_census_default():
    return 0 

def xgboost_census_gs():
    return 0 

def xgboost_census_rs():
    return 0 

def xgboost_census_bo():
    return 0 

def xgboost_census_bohb():
    return 0


