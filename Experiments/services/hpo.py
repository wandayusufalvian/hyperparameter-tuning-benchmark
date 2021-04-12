from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
#from hpbandster_sklearn import HpBandSterSearchCV
#from hpbandster.optimizers import BOHB

def random_search(X,y,model,parameter,iterasi,evalscore,seed):
    auc=[]
    std=[]
    prmtr=[]

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

    auc.append(search.cv_results_['mean_test_score'])
    std.append(search.cv_results_['std_test_score'])
    prmtr.append(search.cv_results_['params'])

    best_index=search.best_index_
    
    return auc,std,prmtr,best_index

def grid_search(X,y,model,parameter,evalscore):
    auc=[]
    std=[]
    prmtr=[]

    search=GridSearchCV(
                model,
                parameter,
                scoring = evalscore,
                n_jobs = -1,
                cv = 5,
                verbose= 10
    )
    
    search.fit(X,y)

    auc.append(search.cv_results_['mean_test_score'])
    std.append(search.cv_results_['std_test_score'])
    prmtr.append(search.cv_results_['params'])

    best_index=search.best_index_
    
    return auc,std,prmtr,best_index

def bayes_opt(X,y,model,parameter,iterasi,evalscore,seed):
    auc=[]
    std=[]
    prmtr=[]

    search= BayesSearchCV(
                   model, 
                   parameter, 
                   n_jobs=-1, 
                   n_iter=iterasi,
                   scoring=evalscore,
                   cv=5,
                   random_state=seed,
                   verbose= 0,
                   iid=True 
    )
    
    search.fit(X,y)

    auc.append(search.cv_results_['mean_test_score'])
    std.append(search.cv_results_['std_test_score'])
    prmtr.append(search.cv_results_['params'])

    best_index=search.best_index_
    
    return auc,std,prmtr,best_index

