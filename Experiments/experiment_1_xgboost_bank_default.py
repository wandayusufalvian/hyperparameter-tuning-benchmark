import services.optimize as opt
import services.gbdt as gbdt 
import services.data as data

iterasi=10

for i in range(0,iterasi):
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    hasil=opt.default_hyperparameter(X,y,model)
    print("Iterasi ke-"+str(i+1)+"\n")
    data.simpan_hasil_hpc(hasil)