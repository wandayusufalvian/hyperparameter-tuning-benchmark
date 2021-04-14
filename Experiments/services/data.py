import os
import pandas as pd 
import numpy as np

def baca_data_bank():
    X_path=os.path.join(os.getcwd(),"dataset-ready","X-bank.csv")
    y_path=os.path.join(os.getcwd(),"dataset-ready","y-bank.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y

def baca_data_credit():
    X_path=os.path.join(os.getcwd(),"dataset-ready","X-credit.csv")
    y_path=os.path.join(os.getcwd(),"dataset-ready","y-credit.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y

def baca_data_census():
    X_path=os.path.join(os.getcwd(),"dataset-ready","X-census.csv")
    y_path=os.path.join(os.getcwd(),"dataset-ready","y-census.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y

def simpan_hasil(name,hasil):
    save_path=os.path.join(os.getcwd(),"results",name+".txt")
    # untuk default hyperparameter
    if len(hasil)==2:
        f=open(save_path,'w')
        kunci=hasil[0].keys()
        for i in kunci:
            f.write(i+" : "+str([hasil[0][i]])+"\n")
        f.write("rata-rata test score : "+str(hasil[0]["test_score"].mean())+"\n")
        f.write("std test score : "+str(hasil[0]["test_score"].std())+"\n")
        f.write("total time: "+str(hasil[1])+" seconds")
        
        f.close()
    # untuk optimized hyperparameter
    else:
        f=open(save_path,'w')
        kunci=hasil[0].keys()
        best_index=hasil[1]
        for i in kunci:
            f.write(i+" : "+str([hasil[0][i]])+"\n")
        f.write("best_index: "+str(best_index)+"\n")
        f.write("best_auc: "+str(hasil[0]['mean_test_score'][best_index])+"\n")
        f.write("best_std: "+str(hasil[0]['std_test_score'][best_index])+"\n")
        f.write("best_param: "+str(hasil[0]['params'][best_index])+"\n")
        f.write("total time : "+str(hasil[2])+" seconds")
        f.close()
