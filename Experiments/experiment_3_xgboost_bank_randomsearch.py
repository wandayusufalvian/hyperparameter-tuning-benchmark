import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

seeds=[42]

for seed in seeds:
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    parameter=hyper.hyper_xgboost_rs()
    hasil=opt.optimized_random_search(X,y,model,parameter,seed)
    print("=====================================Seed = "+str(seed+1)+"======================================================"+"\n")
    data.simpan_hasil_hpc(hasil)

