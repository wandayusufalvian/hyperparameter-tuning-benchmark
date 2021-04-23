import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

def exp_1():
    # experiment_1_xgboost_bank_default 
    iterasi=10
    for i in range(0,iterasi):
        X,y=data.baca_data_bank()
        model=gbdt.xgboost_model()
        hasil=opt.default_hyperparameter(X,y,model)
        print("Iterasi ke-"+str(i+1)+"\n")
        #data.simpan_hasil_hpc(hasil)
        data.simpan_hasil_local("experiment_1_2",hasil)

def exp_2():
    # experiment_2_xgboost_bank_gridsearch
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    parameter=hyper.hyper_xgboost_gs()
    hasil=opt.optimized_grid_search(X,y,model,parameter)
    data.simpan_hasil_hpc(hasil)

def exp_3():
    seeds=[1,12,22,32,42,52,62,72,82,92]

    for seed in seeds:
        X,y=data.baca_data_bank()
        model=gbdt.xgboost_model()
        parameter=hyper.hyper_xgboost_rs()
        hasil=opt.optimized_random_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_4():
    # experiment_4_xgboost_bank_bayessearch
    seeds=[1,12,22,32,42,52,62,72,82,92]

    for seed in seeds:
        X,y=data.baca_data_bank()
        model=gbdt.xgboost_model()
        parameter=hyper.hyper_xgboost_bo()
        hasil=opt.optimized_bayesian_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_5():
    iterations=[3]
    resources='n_samples'
    resources_type=int 
    min_resources=None
    max_resources=None

    # for iter in iterations:git p
    #     X,y=data.baca_data_bank()
    #     model=gbdt.xgboost_model()
    #     parameter=hyper.hyper_xgboost_bohb()
    #     hasil=opt.optimized_bohb(X,y,model,parameter,iter,resources,resources_type,min_resources,max_resources)
    #     print("=====================================Iterations = "+str(iter)+"======================================================"+"\n")
    #     data.simpan_hasil_hpc(hasil)

    # iterations=[10]
    # resources='n_samples'
    # resources_type=int 
    # min_resources=0.3
    # max_resources=1.0

    # for iter in iterations:
    #     X,y=data.baca_data_bank()
    #     model=gbdt.xgboost_model()
    #     parameter=hyper.hyper_xgboost_bohb()
    #     hasil=opt.optimized_bohb(X,y,model,parameter,iter,resources,resources_type,min_resources,max_resources)
    #     print("=====================================Iterations = "+str(iter)+"======================================================"+"\n")
    #     data.simpan_hasil_hpc(hasil)

def exp_6():
    iterasi=10

    for i in range(0,iterasi):
        X,y=data.baca_data_bank()
        model=gbdt.lightgbm_model()
        hasil=opt.default_hyperparameter(X,y,model)
        print("Iterasi ke-"+str(i+1)+"\n")
        #data.simpan_hasil_hpc(hasil)
        data.simpan_hasil_local("experiment_6_2",hasil)

def exp_7():
    X,y=data.baca_data_bank()
    model=gbdt.lightgbm_model()
    parameter=hyper.hyper_lightgbm_gs_2()
    hasil=opt.optimized_grid_search(X,y,model,parameter)
    data.simpan_hasil_hpc(hasil)

def exp_8():
    seeds=[1,12,22,32,42,52,62,72,82,92]
    for seed in seeds:
        X,y=data.baca_data_bank()
        model=gbdt.lightgbm_model()
        parameter=hyper.hyper_lightgbm_rs_2()
        hasil=opt.optimized_random_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_9():
    seeds=[1,12,22,32,42,52,62,72,82,92]

    for seed in seeds:
        X,y=data.baca_data_bank()
        model=gbdt.lightgbm_model()
        parameter=hyper.hyper_lightgbm_bo_2()
        hasil=opt.optimized_bayesian_search_2(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_10():
    return 0

def exp_11():
    # experiment_11_catboost_bank_default
    iterasi=10
    for i in range(0,iterasi):
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model()
        hasil=opt.default_hyperparameter(X,y,model)
        print("======================================Iterasi ke-"+str(i+1)+"================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_12():
    # experiment_12_catboost_bank_gridsearch
    X,y=data.baca_data_bank_catboost()
    model=gbdt.catboost_model()
    parameter=hyper.hyper_catboost_gs()
    hasil=opt.optimized_grid_search(X,y,model,parameter)
    data.simpan_hasil_hpc(hasil)


def exp_13_1():
    # experiment_13_catboost_bank_randomsearch
    seeds=[1,12,22,32,42,52,62,72,82,92]
    for seed in seeds:
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model()
        parameter=hyper.hyper_catboost_rs()
        hasil=opt.optimized_random_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_13_2():
    # experiment_13_catboost_bank_randomsearch
    seeds=[1,12,22,32,42,52,62,72,82,92]
    for seed in seeds:
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model()
        parameter=hyper.hyper_catboost_rs()
        hasil=opt.optimized_random_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_14():
    # experiment_14_catboost_bank_bayessearch
    seeds=[1,12,22,32,42,52,62,72,82,92]
    for seed in seeds:
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model()
        parameter=hyper.hyper_catboost_bo()
        hasil=opt.optimized_bayesian_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_15():
    return 0