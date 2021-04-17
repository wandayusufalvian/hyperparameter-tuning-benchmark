import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

seed=[1]

for i in seed:
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    parameter=hyper.hyper_xgboost_bo()
    hasil=opt.optimized_random_search(X,y,model,parameter,seed)
    print("=====================================Seed = "+str(i+1)+"======================================================"+"\n")
    data.simpan_hasil_hpc(hasil)

