#%%
'''import libraries'''
import pandas as pd
import numpy as np
from service import *

#%%
'''baca file'''  
data_bank=pd.read_csv(r"/home/yusuf/hyperparameter-tuning-benchmark/Data_Preprocessing/dataset-raw/bank-full.csv",sep=';')
# data_bank.info()

#%%
'''hapus kolom duration'''
data_bank=data_bank.drop(['duration'],axis=1)
# berdasarkan https://archive.ics.uci.edu/ml/datasets/Bank+Marketing, disarankan duration dihapus 

#%% 
'''banyak nilai kategori di tiap fitur kategorikal''' 
# banyak_tiap_kategori(data_bank)

#%% 
'''ubah nilai unknown menjadi nilai yang paling sering muncul'''
data_bank['job']=data_bank['job'].replace(['unknown'],'blue-collar')
data_bank['education']=data_bank['education'].replace(['unknown'],'secondary')
data_bank['contact']=data_bank['contact'].replace(['unknown'],'cellular')

#%%
'''hapus kolom poutcome'''
data_bank=data_bank.drop(['poutcome'],axis=1)

#%%
'''pengelompokan nilai kategori pada fitur job'''
data_bank['job']=data_bank['job'].replace(['management','entrepreneur'],'high')
data_bank['job']=data_bank['job'].replace(['blue-collar','technician','services','self-employed'],'mid')
data_bank['job']=data_bank['job'].replace(['unemployed','housemaid','student','admin.','retired'],'low')

'''pengelompokan nilai kategori pada fitur month'''
data_bank['month']=data_bank['month'].replace(['jan','feb','mar','apr','may','jun'],'semester-1')
data_bank['month']=data_bank['month'].replace(['jul','aug','sep','oct','nov','dec'],'semester-2')

#%%
## untuk catboost ini tidak perlu dilakukan...
'''label encoding pada fitur education'''
kategori=['primary','secondary','tertiary']
label=[1,2,3]
data_bank['education']=data_bank['education'].replace(kategori,label)

#%%
'''dimensi data sebelum transformasi'''
X,y=pisah_x_y(data_bank)
# print("before transform")
# print("y= ",y.shape)
# print("X= ",X.shape)
#%%
'''transformasi data'''
numerical_minmax=['age','balance','day','campaign','pdays','previous']
categorical_onehot=['job','marital','default','housing','loan','contact','month']
y=transform_kelas(y)
#X=transform_fitur(X,numerical_minmax,categorical_onehot) # xgboost dan lightgbm
X=transform_fitur_catboost(X,numerical_minmax) # catboost
X=pd.DataFrame(X)
y=pd.DataFrame(y)
# pembulatan hasil dari min max scaler => ternyata gk ada bedanya dibulatkan dengan tidak 
#float_col=[0,1,2,3,4,5]
#pembulatan(X,float_col)





#%%
print("dimensi data setelah transformasi")
print("y= ",y.shape)
print("X= ",X.shape)

#%%
'''export data'''
ekspor_data(X,y,"X-bank-catboost2","y-bank")

