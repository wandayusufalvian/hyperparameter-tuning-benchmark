# experiment_8_lightgbm_bank_randomsearch
import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

seeds=[1,12,22,32,42,52,62,72,82,92]

for seed in seeds:
    X,y=data.baca_data_bank()
    model=gbdt.lightgbm_model()
    parameter=hyper.hyper_lightgbm_rs()
    hasil=opt.optimized_random_search(X,y,model,parameter,seed)
    print("=====================================Seed = "+str(seed)+"======================================================"+"\n")
    data.simpan_hasil_hpc(hasil)

