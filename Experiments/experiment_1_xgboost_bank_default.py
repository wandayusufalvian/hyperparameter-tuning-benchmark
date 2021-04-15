from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import time

skf=StratifiedKFold(n_splits=5)
eval_method="roc_auc"
iterasi=200

def default_hyperparameter(X,y,model):
    start_time = time.time()
    cv_results = cross_validate(model,X,y,cv=skf,scoring=eval_method)
    end_time = time.time()
    list_hasil=[cv_results,end_time-start_time]
    return list_hasil

from xgboost import XGBClassifier
def xgboost_model():
    return XGBClassifier(verbosity = 0,use_label_encoder =False)

import os
import pandas as pd 
import numpy as np

def baca_data_bank():
    X_path=os.path.join(os.getcwd(),"dataset-ready","X-bank.csv")
    y_path=os.path.join(os.getcwd(),"dataset-ready","y-bank.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y
def simpan_hasil_hpc(hasil):
    # untuk default hyperparameter
    if len(hasil)==2:
        kunci=hasil[0].keys()
        for i in kunci:
            print(i+" : "+str([hasil[0][i]])+"\n")
        print("rata-rata test score : "+str(hasil[0]["test_score"].mean())+"\n")
        print("std test score : "+str(hasil[0]["test_score"].std())+"\n")
        print("total time: "+str(hasil[1])+" seconds"+"\n\n")

    # untuk optimized hyperparameter
    else:
        kunci=hasil[0].keys()
        best_index=hasil[1]
        for i in kunci:
            print(i+" : "+str([hasil[0][i]])+"\n")
        print("best_index: "+str(best_index)+"\n")
        print("best_auc: "+str(hasil[0]['mean_test_score'][best_index])+"\n")
        print("best_std: "+str(hasil[0]['std_test_score'][best_index])+"\n")
        print("best_param: "+str(hasil[0]['params'][best_index])+"\n")
        print("total time : "+str(hasil[2])+" seconds"+"\n\n")
iterasi=1

for i in range(0,iterasi):
    X,y=baca_data_bank()
    model=xgboost_model()
    hasil=default_hyperparameter(X,y,model)
    print("Iterasi ke-"+str(i+1)+"\n")
    simpan_hasil_hpc(hasil)