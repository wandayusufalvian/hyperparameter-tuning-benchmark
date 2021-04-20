import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

X,y=data.baca_data_bank()
model=gbdt.xgboost_model()
parameter=hyper.hyper_xgboost_gs()
hasil=opt.optimized_grid_search(X,y,model,parameter)
data.simpan_hasil_hpc(hasil)