from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
#from hpbandster_sklearn import HpBandSterSearchCV
#from hpbandster.optimizers import BOHB

def random_search(X,y,model,parameter,iterasi,evalscore,seed):
    auc=[]
    std=[]

    random_search=RandomizedSearchCV(
                   model,
                   parameter, 
                   n_iter=iterasi, 
                   scoring=evalscore, 
                   n_jobs=-1, 
                   cv=5, 
                   random_state=seed,
                   verbose= 1)
    
    random_search.fit(X,y)

    best_index=random_search.best_index_
    auc.append(random_search.cv_results_['mean_test_score'])
    std.append(random_search.cv_results_['std_test_score'])
    best_param=random_search.best_params_
    
    return auc,std,best_index,best_param



