import os
import pandas as pd 
import numpy as np

def read_data_bank():
    X_path=os.path.join(os.getcwd(),"Experiments","dataset-ready","X-bank.csv")
    y_path=os.path.join(os.getcwd(),"Experiments","dataset-ready","y-bank.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y

def read_data_credit():
    X_path=os.path.join(os.getcwd(),"Experiments","dataset-ready","X-credit.csv")
    y_path=os.path.join(os.getcwd(),"Experiments","dataset-ready","y-credit.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y

def read_data_census():
    X_path=os.path.join(os.getcwd(),"Experiments","dataset-ready","X-census.csv")
    y_path=os.path.join(os.getcwd(),"Experiments","dataset-ready","y-census.csv")
    X=np.genfromtxt(X_path, delimiter=',')
    y=np.genfromtxt(y_path, delimiter=',')
    return X,y

