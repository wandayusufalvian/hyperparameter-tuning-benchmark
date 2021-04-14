import services.optimize as opt
import services.gbdt as gbdt 
import services.data as data

nama_file=["xgboost_bank_default_1","xgboost_bank_default_2","xgboost_bank_default_3","xgboost_bank_default_4","xgboost_bank_default_5"]

for i in nama_file:
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    hasil=opt.default_hyperparameter(X,y,model)
    data.simpan_hasil(i,hasil)