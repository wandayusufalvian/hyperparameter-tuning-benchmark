#%%
'''import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from service import *

#%%
'''baca file'''  
data_credit=pd.read_csv(r"/home/yusuf/hyperparameter-tuning-benchmark/Data_Preprocessing/dataset-raw/credit_card.csv")
data_credit=data_credit.drop(['ID'],axis=1)
data_credit.info()

# %%
'''pengelompokan nilai kategori pada fitur education'''
data_credit['EDUCATION']=data_credit['EDUCATION'].replace([5,6,0],4)

# %%
'''pengelompokan nilai kategori pada fitur marriage'''
data_credit['MARRIAGE']=data_credit['MARRIAGE'].replace([0],3)
print(data_credit['MARRIAGE'].value_counts())

#%%
X,y=pisah_x_y(data_credit)

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

