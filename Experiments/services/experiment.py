from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import time

def default_hyperparameter(model,X,y):
    start_time = time.time()

    skf=StratifiedKFold(n_splits=5)
    results=cross_val_score(model,X,y,cv=skf,scoring="roc_auc")

    end_time = time.time()

    list_hasil=[str(round(results.mean(),4)),str(round(results.std(),4)),str(end_time-start_time)]
    return list_hasil
 

def optimized_random_search(X,y,model,parameter,iterasi,evalscore,seed):
    auc=[]
    std=[]
    prmtr=[]

    start_time = time.time()
    search=RandomizedSearchCV(
                   model,
                   parameter, 
                   n_iter=iterasi, 
                   scoring=evalscore, 
                   n_jobs=-1, 
                   cv=5, 
                   random_state=seed,
                   verbose= 10)
    search.fit(X,y)
    end_time = time.time()

    auc.append(search.cv_results_['mean_test_score'])
    std.append(search.cv_results_['std_test_score'])
    prmtr.append(search.cv_results_['params'])

    best_index=search.best_index_

    best_auc=auc[0][best_index]
    best_std=std[0][best_index]
    best_param=prmtr[0][best_index]

    temp=[auc,std,prmtr,best_index,best_auc,best_std,best_param,end_time-start_time]

    list_hasil=[str(i) for i in temp]
    return list_hasil
