from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import time

skf=StratifiedKFold(n_splits=5)
eval_method="roc_auc"
iterasi=200

def default_hyperparameter(X,y,model):
    start_time = time.time()
    cv_results = cross_validate(model,X,y,cv=skf,scoring=eval_method)
    end_time = time.time()
    list_hasil=[cv_results,end_time-start_time]
    return list_hasil

def optimized_grid_search(X,y,model,parameter):
    start_time = time.time()
    search=GridSearchCV(
                model,
                parameter,
                scoring = eval_method,
                n_jobs = -1,
                cv = 5,
                verbose= 0
    )
    search.fit(X,y)
    end_time = time.time()

    return [search.cv_results_,search.best_index_,end_time-start_time]

def optimized_random_search(X,y,model,parameter,seed):
    start_time = time.time()
    search=RandomizedSearchCV(
                   model,
                   parameter, 
                   n_iter=iterasi, 
                   scoring=eval_method, 
                   n_jobs=-1, 
                   cv=5, 
                   random_state=seed,
                   verbose= 0)
    search.fit(X,y)
    end_time = time.time()

    return [search.cv_results_,search.best_index_,end_time-start_time]

def optimized_bayesian_search(X,y,model,parameter,seed):
    start_time = time.time()
    search= BayesSearchCV(
                   model, 
                   parameter, 
                   n_jobs=-1, 
                   n_iter=iterasi,
                   scoring=eval_method,
                   cv=5,
                   random_state=seed,
                   verbose= 0,
                   iid=True 
    )
    search.fit(X,y)
    end_time = time.time()

    return [search.cv_results_,search.best_index_,end_time-start_time]
