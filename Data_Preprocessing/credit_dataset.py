#%%
'''import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from service import *

#%%
'''read files and explore data'''  
data_credit=pd.read_csv(r"/home/yusuf/hyperparameter-tuning-benchmark/Data_Preprocessing/dataset-raw/credit_card.csv")
data_credit=data_credit.drop(['ID'],axis=1)
#data_credit.info()

# %%
'''grouping education feature'''
print(data_credit['EDUCATION'].unique())
print(data_credit['EDUCATION'].value_counts())
data_credit['EDUCATION']=data_credit['EDUCATION'].replace([5,6,0],4)
print(data_credit['EDUCATION'].value_counts())

# %%
#print("unique values SEX")
#print(data_credit['SEX'].unique())

# %%
'''grouping marriage feature'''
print(data_credit['MARRIAGE'].unique())
print(data_credit['MARRIAGE'].value_counts())
data_credit['MARRIAGE']=data_credit['MARRIAGE'].replace([0],3)
print(data_credit['MARRIAGE'].value_counts())

#%%

print(data_credit['SEX'].value_counts())

#%%
X,y=pisah_x_y(data_credit)
print("before transform")
print("y= ",y.shape)
print("X= ",X.shape)

categorical_onehot=['SEX','MARRIAGE']
numerical_minmax=['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' ]

y=transform_kelas(y)
X=transform_fitur(X,numerical_minmax,categorical_onehot)

print("after transform")
print("y= ",y.shape)
print("X= ",X.shape)

#%%
'''export data'''
ekspor_data(X,y,"X-credit","y-credit")
# %%
