import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

iterations=[10]

for iter in iterations:
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    parameter=hyper.hyper_xgboost_bohb()
    hasil=opt.optimized_bohb(X,y,model,parameter,iter)
    print("=====================================Seed = "+str(iter)+"======================================================"+"\n")
    data.simpan_hasil_hpc(hasil)

