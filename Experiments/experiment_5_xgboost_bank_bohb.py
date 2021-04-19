import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

iterations=[5]
resources='n_samples'
# jika resources n_estimators maka pasti harus set nilai max_resources 
# untuk n_samples sepertinya juga harus divariasikan nilai max_resources karena
# ketika run sekali menggunakan n_samples dan default max_resources(all)=>telralu lama 
for iter in iterations:
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    parameter=hyper.hyper_xgboost_bohb()
    hasil=opt.optimized_bohb(X,y,model,parameter,iter,resources,0.3,1)
    print("=====================================Seed = "+str(iter)+"======================================================"+"\n")
    data.simpan_hasil_hpc(hasil)

