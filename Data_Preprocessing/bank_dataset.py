#%%
'''import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer 

from service import *


#%%
'''read files and explore data'''  
data_bank=pd.read_csv(r"/home/yusuf/hyperparameter-tuning-benchmark/Data_Preprocessing/dataset-raw/bank-full.csv",sep=';')

#%%
'''delete duration column'''
data_bank=data_bank.drop(['duration'],axis=1)
# berdasarkan https://archive.ics.uci.edu/ml/datasets/Bank+Marketing, disarankan duration dihapus 

#%%
'''show features'''
data_bank.info()

#%% 
'''show categorical values''' 
# show_values(data_bank)

#%% 
'''change unknown values => most frequent values'''
# job = blue-collar 
# education = secondary 
# contact = cellular 
data_bank['job']=data_bank['job'].replace(['unknown'],'blue-collar')
data_bank['education']=data_bank['education'].replace(['unknown'],'secondary')
data_bank['contact']=data_bank['contact'].replace(['unknown'],'cellular')

#%%
'''calculate percentage of unknown data'''
#print("job : ",288*100/45211)
#print("education : ",1857*100/45211)
#print("contact : ",13020*100/45211)
#print("poutcome : ",36959*100/45211)

#%%
'''delete poutcome column'''
data_bank=data_bank.drop(['poutcome'],axis=1)

#%%
'''categories quantity'''
#X,y=split_x_y(data_bank)
#count_categories(X)

#%%
'''group categories value in job feature'''
data_bank['job']=data_bank['job'].replace(['management','entrepreneur'],'high')
data_bank['job']=data_bank['job'].replace(['blue-collar','technician','services','self-employed'],'mid')
data_bank['job']=data_bank['job'].replace(['unemployed','housemaid','student','admin.','retired'],'low')

'''group categories value in month feature'''
data_bank['month']=data_bank['month'].replace(['jan','feb','mar','apr','may','jun'],'semester-1')
data_bank['month']=data_bank['month'].replace(['jul','aug','sep','oct','nov','dec'],'semester-2')

#show_values(data_bank)

#%%
'''transform and normalize data'''
X,y=split_x_y(data_bank)
print("before transform")
print("y= ",y.shape)
print("X= ",X.shape)
X,y=transform_data(X,y)
print("after transform")
print("y= ",y.shape)
print("X= ",X.shape)

#%%
'''export data'''
ekspor_data(X,y,"X-bank","y-bank")