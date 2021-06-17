import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import math


def calc_unique(X,koloms):
    '''menghitung banyak nilai unik untuk fitur pada list koloms'''
    for i,k in enumerate(koloms):
        print(i+1,". ",k,'----------------------\n')
        print("nilai unik: ",len(X[k].unique()),'\n')
        print(X[k].value_counts(),'\n')


def split_x_y(data):
    '''pisah data jadi X dan y'''
    y=data.iloc[:,-1:]
    X=data.iloc[:,0:-1]
    return X,y

def transform_kelas(y):
    y=y.to_numpy().ravel()
    y=LabelEncoder().fit_transform(y)
    return y

def transform_fitur(X,numerical_col,categorical_onehot):
    t = [('cat', OneHotEncoder(sparse=False), categorical_onehot),('num', MinMaxScaler(), numerical_col)]
    col_transform = ColumnTransformer(transformers=t,remainder='passthrough')
    X=col_transform.fit_transform(X)
    return X

def transform_fitur_onehot(X,categorical_onehot):
    t = [('cat', OneHotEncoder(sparse=False), categorical_onehot)]
    col_transform = ColumnTransformer(transformers=t,remainder='passthrough')
    X=col_transform.fit_transform(X)
    return X

def transform_fitur_minmaxscaler(X,numerical_col):
    t = [('num', MinMaxScaler(), numerical_col)]
    col_transform = ColumnTransformer(transformers=t,remainder='passthrough')
    X=col_transform.fit_transform(X)
    return X

def pembulatan(X,float_col):
    for i in float_col:
        X[i]=X[i].astype(float)
        X[i]=X[i].round(3)

def transform_fitur_catboost(X,numerical_col):
    t = [('num', MinMaxScaler(), numerical_col)]
    col_transform = ColumnTransformer(transformers=t,remainder='passthrough')
    X=col_transform.fit_transform(X)
    return X

def ekspor_data(X,y,nama_X,nama_y):
    X.to_csv('dataset-ready/'+nama_X+'.csv',index=False,header=None)
    y.to_csv('dataset-ready/'+nama_y+'.csv',index=False,header=None)

def hitung_kategori(X):
    categorical_ix = X.select_dtypes(include=['object']).columns 
    sum=0
    for i in categorical_ix:
        print(i,"=",len(X[i].unique())," kategori")
        sum=sum+len(X[i].unique())
    print("\n total kategori= ",sum)


def balance_or_not(banyak_kelas, list_jumlah):
    '''
    menentukan apakah suatu data kategorikal balance atau tidak
    sumber : https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
    '''
    n=sum(list_jumlah)
    H=0 
    for i in list_jumlah:
        H=H+(i/n)*math.log(i/n)
    print(H)
    return -H/math.log(banyak_kelas)

