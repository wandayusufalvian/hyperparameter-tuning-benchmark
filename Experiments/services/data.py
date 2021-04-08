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


def simpan_hasil(name,list_hasil):
    save_path=os.path.join(os.getcwd(),"results",name+".txt")
    # untuk default hyperparameter 
    if len(list_hasil)==3:
        f=open(save_path,'w')
        f.write("AUC: "+list_hasil[0]+"\n")
        f.write("std: "+list_hasil[1]+"\n")
        f.write("time: "+list_hasil[2]+" seconds")
        f.close()
    # untuk yang menggunakan hpo 
    else:
        f=open(save_path,'w')
        f.write("AUC: "+list_hasil[0]+"\n")
        f.write("std: "+list_hasil[1]+"\n")
        f.write("params: "+list_hasil[2]+"\n")
        f.write("best_index: "+list_hasil[3]+"\n")
        f.write("best_auc: "+list_hasil[4]+"\n")
        f.write("best_std: "+list_hasil[5]+"\n")
        f.write("best_param: "+list_hasil[6]+"\n")
        f.write("time: "+list_hasil[7]+" seconds")
        f.close()
    

