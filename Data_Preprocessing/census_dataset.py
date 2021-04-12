#%%
'''import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from service import *

#%%
'''baca data'''  
data_census=pd.read_csv(r"/home/yusuf/hyperparameter-tuning-benchmark/Data_Preprocessing/dataset-raw/census_income.csv")
data_census.info()

#%%
'''banyak nilai kategori di tiap fitur kategorikal''' 
banyak_tiap_kategori(data_census)

#%%
# '''hitung persentase data dengan nilai '?' '''
# print("workclass : ", 1836*100/32561)
# print("occupation : ", 1843*100/32561)
# print("native.country : ", 583*100/32561)
# %%
'''ubah nilai '?' menjadi nilai yang paling sering muncul pada tiap fitur kategorikal'''
data_census['workclass']=data_census['workclass'].replace(['?'],'Private')
data_census['occupation']=data_census['occupation'].replace(['?'],'Prof-specialty')
data_census['native.country']=data_census['native.country'].replace(['?'],'United-States')

#%%
'''
cek banyak nilai pada data kategorikal 
setelah ubah nilai yg tidak relevan 
'''
banyak_tiap_kategori(data_census)

# %%
'''
hitung banyak nilai kategorikal di taip data kategorikal 
'''
hitung_kategori(data_census)

# %%
'''transformasi nilai kategori pada fitur education'''
di={
    'Doctorate':15,
    'Masters':14,
    'Bachelors':13, 
    'Assoc-voc':12, 
    'Assoc-acdm':12,
    'Some-college':12,
    'Prof-school':12,
    'HS-grad':11,
    '12th':10, 
    '11th':9,
    '10th':8,
    '9th':7, 
    '7th-8th':6,
    '5th-6th':5,
    '1st-4th':4,
    'Preschool':3
}
data_census['education']=data_census['education'].map(di)
data_census['education'].value_counts()

'''pengelompokan nilai kategori pada fitur kategorikal'''
#%%
data_census['workclass']=data_census['workclass'].replace(['Self-emp-not-inc', 'Self-emp-inc' ],'Self-emp')
data_census['workclass']=data_census['workclass'].replace(['Local-gov', 'State-gov', 'Federal-gov'],'Gov-emp')
data_census['workclass']=data_census['workclass'].replace(['Without-pay','Never-worked'],'Others')

data_census['marital.status']=data_census['marital.status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'],'Married')
data_census['marital.status']=data_census['marital.status'].replace(['Divorced', 'Separated', 'Widowed'],'Others')

data_census['occupation']=data_census['occupation'].replace(['Prof-specialty', 'Craft-repair' ],'Professional')
data_census['occupation']=data_census['occupation'].replace(['Exec-managerial', 'Adm-clerical', 'Sales', 'Armed-Forces', 'Tech-support', 'Machine-op-inspct'],'White-collar')
data_census['occupation']=data_census['occupation'].replace(['Other-service', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Protective-serv', 'Priv-house-serv'],'Blue-collar')


data_census['relationship']=data_census['relationship'].replace(['Not-in-family','Unmarried','Other-relative','Own-child'],'Others')


data_census['race']=data_census['race'].replace(['Amer-Indian-Eskimo','Asian-Pac-Islander','Other'],'Others')


data_census['native.country']=data_census['native.country'].replace(['Cambodia', 'China', 'Hong', 'Laos', 'Thailand', 'Japan', 'Taiwan', 'Vietnam', 'India', 'Iran', 'Philippines', 'South'],'Asia')
data_census['native.country']=data_census['native.country'].replace(['Cuba', 'Guatemala', 'Jamaica', 'Nicaragua', 'Puerto-Rico', 'Dominican-Republic', 'El-Salvador', 'Haiti', 'Honduras', 'Mexico', 'Trinadad&Tobago', 'Ecuador', 'Peru', 'Columbia' ],'South-America')
data_census['native.country']=data_census['native.country'].replace(['England', 'Germany', 'Holand-Netherlands', 'Ireland', 'France', 'Greece', 'Italy', 'Portugal', 'Scotland', 'Poland', 'Yugoslavia', 'Hungary'],'Europe')
data_census['native.country']=data_census['native.country'].replace(['United-States', 'Outlying-US(Guam-USVI-etc)', 'Canada'],'North-America')

#%%
'''
setelah di grouping,
check ulang nilai kategorinya, pastikan sesuai
'''
banyak_tiap_kategori(data_census)

#%%
'''
split data 
'''
X,y=pisah_x_y(data_census)


# %%
'''
sebelum melakukan one hot encoding
hitung jumlah kategori tiap fitur
yang tipe datanya object 
'''
hitung_kategori(X)
#%%
'''
dimensi data setelah one hot 
encoding (hitung manual)
'''
print('dimensi= ',22+7)

# %%
print("before transform")
print("y= ",y.shape)
print("X= ",X.shape)


numerical_minmax=['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss']

categorical_onehot=['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
y=transform_kelas(y)
X=transform_fitur(X,numerical_minmax,categorical_onehot)


print("after transform")
print("y= ",y.shape)
print("X= ",X.shape)

#%%
'''export data'''
ekspor_data(X,y,"X-census","y-census")
# %%
