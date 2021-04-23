from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.callbacks import DeltaXStopper
#from hpbandster_sklearn import HpBandSterSearchCV
import time

skf=StratifiedKFold(n_splits=5)
eval_method="roc_auc"

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
                   n_iter=200, 
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
                   n_iter=200,
                   scoring=eval_method,
                   cv=5,
                   random_state=seed,
                   verbose= 0
    )
    search.fit(X,y)
    end_time = time.time()

    return [search.cv_results_,search.best_index_,end_time-start_time]

def optimized_bayesian_search_2(X,y,model,parameter,seed):
    start_time = time.time()
    search= BayesSearchCV(
                   model, 
                   parameter, 
                   n_jobs=-1, 
                   n_iter=200,
                   scoring=eval_method,
                   cv=5,
                   random_state=seed,
                   verbose= 0,
                   optimizer_kwargs={'base_estimator': 'GP'}
    )
    search.fit(X,y,callback=DeltaXStopper(0.0001))
    end_time = time.time()

    return [search.cv_results_,search.best_index_,end_time-start_time]

# def optimized_bohb(X,y,model,parameter,iter,rsc,rsc_type,min_rsc,max_rsc):
#     start_time = time.time()
#     search= HpBandSterSearchCV(
#                    model, 
#                    parameter, 
#                    n_jobs=-1, 
#                    n_iter=iter, # beda dengan iterasi pada random dan bayes search
#                    resource_name=rsc, # n_samples or n_estimator
#                    resource_type=rsc_type,
#                    scoring=eval_method,
#                    min_budget=min_rsc,
#                    max_budget=max_rsc,
#                    cv=5,
#                    verbose= 0
#     )
#     search.fit(X,y)
#     end_time = time.time()

#     return [search.cv_results_,search.best_index_,end_time-start_time]