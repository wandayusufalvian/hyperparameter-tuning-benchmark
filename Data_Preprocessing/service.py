import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

def banyak_tiap_kategori(X):
    cat_df=X.select_dtypes(include='object')
    cat_cols=cat_df.columns
    for i in cat_cols:
        print(i,'\n')
        print(X[i].value_counts(),'\n')

def pisah_x_y(data):
    y=data.iloc[:,-1:]
    X=data.iloc[:,0:-1]
    return X,y

def hitung_kategori(X):
    categorical_ix = X.select_dtypes(include=['object']).columns 
    sum=0
    for i in categorical_ix:
        print(i,"=",len(X[i].unique())," categories")
        sum=sum+len(X[i].unique())
    print("\n total categories= ",sum)

def transform_kelas(y):
    y=y.to_numpy().ravel()
    y=LabelEncoder().fit_transform(y)
    return y

def transform_fitur(X,numerical_col,categorical_onehot):
    t = [('cat', OneHotEncoder(sparse=False), categorical_onehot),('num', MinMaxScaler(), numerical_col)]
    col_transform = ColumnTransformer(transformers=t,remainder='passthrough')
    X=col_transform.fit_transform(X)
    return X

def ekspor_data(X,y,nama_X,nama_y):
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    X.to_csv('dataset-ready/'+nama_X+'.csv',index=False,header=None)
    y.to_csv('dataset-ready/'+nama_y+'.csv',index=False,header=None)