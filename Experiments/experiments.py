import services.reader_writer as reader_writer
import services.models as gbdt
import services.optimizer as opt 
import services.hyperparameter as hyper

def exp_default(**kwargs):
    '''
    Parameters
    ----------
    model = xgboost, lightgbm, atau catboost
    dataset = dataset 1,2, atau 3
    iterasi = di-run berapa kali. hasil tidak akan berubah, hanya untuk lihat waktu eksekusi-nya
    file_name = nama txt file untuk simpan hasil eksperimen. jika run di HPC, file_name harus ditulis ulang pada .sh file.
    mesin = 1 jika run locally. selain 1 jika ingin run di HPC.  
    '''
    for i in range(0,kwargs["iterasi"]):
        X=kwargs["dataset"][0]
        y=kwargs["dataset"][1]
        hasil=opt.no_optimizer(X,y,kwargs["model"])
        print("Iterasi ke-"+str(i+1)+"\n")
        if kwargs["mesin"]==1:
            reader_writer.simpan_hasil_local(kwargs["file_name"],hasil)  
        else:
            reader_writer.simpan_hasil_hpc(hasil)  

def exp_grid_search(**kwargs):
    '''
    Parameters
    ----------
    model = xgboost, lightgbm, atau catboost
    dataset = dataset 1,2, atau 3
    hyperparameter= dict of hyperparameter list
    file_name = nama txt file untuk simpan hasil eksperimen. jika run di HPC, file_name harus ditulis ulang pada .sh file.
    mesin = 1 jika run locally. selain 1 jika ingin run di HPC.  
    '''
    X=kwargs["dataset"][0]
    y=kwargs["dataset"][1]
    hasil=opt.optimized_grid_search(X,y,kwargs["model"],kwargs["hyperparameter"])
    if kwargs["mesin"]==1:
        reader_writer.simpan_hasil_local(kwargs["file_name"],hasil)  
    else:
        reader_writer.simpan_hasil_hpc(hasil)  

def exp_random_search(**kwargs):
    '''
    Parameters
    ----------
    model = xgboost, lightgbm, atau catboost
    dataset = dataset 1,2, atau 3
    hyperparameter= dict of hyperparameter list
    seeds = list of seed for random behaviour
    file_name = nama txt file untuk simpan hasil eksperimen. jika run di HPC, file_name harus ditulis ulang pada .sh file.
    mesin = 1 jika run locally. selain 1 jika ingin run di HPC.  
    '''
    for seed in kwargs["seeds"]:
        X=kwargs["dataset"][0]
        y=kwargs["dataset"][1]
        hasil=opt.optimized_random_search(X,y,kwargs["model"],kwargs["hyperparameter"],seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        if kwargs["mesin"]==1:
            reader_writer.simpan_hasil_local(kwargs["file_name"],hasil)  
        else:
            reader_writer.simpan_hasil_hpc(hasil) 

def exp_bayes_search():
    pass

def exp_bohb():
    pass 


def exp_3():
    

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
def exp_9_1():
    # run ulang yang seed =22 karena pada exp_9 iterasi tidak sampai 200 sehingga gk fair dengan exp yg lain 
    # solusinya = jangan pakai bayes search yg ada stopping nya 
    seeds=[22]

    for seed in seeds:
        X,y=data.baca_data_bank()
        model=gbdt.lightgbm_model()
        parameter=hyper.hyper_lightgbm_bo_2()
        hasil=opt.optimized_bayesian_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_10():
    return 0

def exp_11():
    # experiment_11_catboost_bank_default
    # pakai model catboost yang beda => harapannya bisa lebih cepat dari exp_11()
    iterasi=10
    for i in range(0,iterasi):
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model_2()
        hasil=opt.default_hyperparameter(X,y,model)
        print("======================================Iterasi ke-"+str(i+1)+"================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_12():
    # experiment_12_catboost_bank_gridsearch
    X,y=data.baca_data_bank_catboost()
    model=gbdt.catboost_model_2()
    parameter=hyper.hyper_catboost_gs_1()
    hasil=opt.optimized_grid_search(X,y,model,parameter)
    data.simpan_hasil_hpc(hasil)

def exp_13():
    # experiment_13_catboost_bank_randomsearch
    seeds=[1,12,22,32,42,52,62,72,82,92]
    for seed in seeds:
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model_2()
        parameter=hyper.hyper_catboost_rs()
        hasil=opt.optimized_random_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_14():
    # experiment_14_catboost_bank_bayessearch
    seeds=[1,12,22,32,42,52,62,72,82,92]
    for seed in seeds:
        X,y=data.baca_data_bank_catboost()
        model=gbdt.catboost_model_2()
        parameter=hyper.hyper_catboost_bo()
        hasil=opt.optimized_bayesian_search(X,y,model,parameter,seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        data.simpan_hasil_hpc(hasil)

def exp_15():
    return 0