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

def exp_bayes_search(**kwargs):
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
        hasil=opt.optimized_bayesian_search(X,y,kwargs["model"],kwargs["hyperparameter"],seed)
        print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
        if kwargs["mesin"]==1:
            reader_writer.simpan_hasil_local(kwargs["file_name"],hasil)  
        else:
            reader_writer.simpan_hasil_hpc(hasil) 

def exp_bohb():
    pass 



