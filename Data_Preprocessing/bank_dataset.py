import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from service import *

'''READ FILES'''  
data_bank=pd.read_csv(r"/mnt/c/Users/Wanda Yusuf Alvian/cs/Code/machine-learning-collections/thesis/Data_Preprocessing/Dataset/bank-full.csv",sep=';')
#data_bank.info()

'''CHECK IRRELEVANT VALUE''' 
# show_values(data_bank)

'''CALCULATE PERCENTAGE OF 'UNKNOWN' DATA'''
# print("job : ",288*100/45211)
# print("education : ",1857*100/45211)
# print("contact : ",13020*100/45211)
# print("poutcome : ",36959*100/45211)

# print(data_bank['contact'])
 

'''DELETE UNKNOWN ROWS IN JOB AND EDUCATION COLUMN'''
data_bank=data_bank.loc[(data_bank.education != 'unknown')]
data_bank=data_bank.loc[(data_bank.job != 'unknown')]
#show_values(data_bank)

'''DELETE POUTCOME COLUMN'''
data_bank=data_bank.drop(['poutcome'],axis=1)
#data_bank.info()
'''DELETE DURATION COLUMN'''
data_bank=data_bank.drop(['duration'],axis=1)
# Important note: this attribute highly affects the 
# output target (e.g., if duration=0 then y='no'). 
# Yet, the duration is not known before a call is performed.

'''CHECK OUTLIER IN CONTINUOUS DATA'''
#plt.boxplot(data_bank['age'])