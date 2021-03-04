import pandas as pd

def show_values(X):
    cat_df=X.select_dtypes(include='object')
    cat_cols=cat_df.columns
    for i in cat_cols:
        print(i,'\n')
        print(X[i].value_counts(),'\n')


def ekspor_data(X,y,nama_X,nama_y):
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    X.to_csv('dataset/'+nama_X+'.csv',index=False)
    y.to_csv('dataset/'+nama_y+'.csv',index=False)

