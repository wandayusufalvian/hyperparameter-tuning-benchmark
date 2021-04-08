from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
#from hpbandster_sklearn import HpBandSterSearchCV
#from hpbandster.optimizers import BOHB

def random_search(X,y,model,parameter,iterasi,evalscore,seed):
    auc=[]
    std=[]
    prmtr=[]

    random_search=RandomizedSearchCV(
                   model,
                   parameter, 
                   n_iter=iterasi, 
                   scoring=evalscore, 
                   n_jobs=-1, 
                   cv=5, 
                   random_state=seed,
                   verbose= 10)
    
    random_search.fit(X,y)

    auc.append(random_search.cv_results_['mean_test_score'])
    std.append(random_search.cv_results_['std_test_score'])
    prmtr.append(random_search.cv_results_['params'])

    best_index=random_search.best_index_
    best_param=random_search.best_params_
    best_auc=auc[best_index]
    best_std=std[best_index]
    
    return auc,std,prmtr,best_index,best_auc,best_std,best_param

def grid_search(X,y,model,parameter,evalscore):
    auc=[]
    std=[]
    prmtr=[]

    grid_search=GridSearchCV(
                model,
                parameter,
                scoring = evalscore,
                n_jobs = -1,
                cv = 5,
                verbose= 10
    )
    
    grid_search.fit(X,y)

    auc.append(grid_search.cv_results_['mean_test_score'])
    std.append(grid_search.cv_results_['std_test_score'])
    prmtr.append(grid_search.cv_results_['params'])

    best_index=grid_search.best_index_
    #best_param=grid_search.best_params_
    
    
    return auc,std,prmtr,best_index

def bayes_opt(X,y,model,parameter,iterasi,evalscore,seed):
    
    auc=[]
    std=[]

    bayes_search= BayesSearchCV(
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
    
    bayes_search.fit(X,y)
    
    auc.append(bayes_search.cv_results_['mean_test_score'])
    std.append(bayes_search.cv_results_['std_test_score'])
    prmtr.append(bayes_search.cv_results_['params'])

    best_index=bayes_search.best_index_
    best_param=bayes_search.best_params_
    best_auc=auc[best_index]
    best_std=std[best_index]
    
    return auc,std,prmtr,best_index,best_auc,best_std,best_param

