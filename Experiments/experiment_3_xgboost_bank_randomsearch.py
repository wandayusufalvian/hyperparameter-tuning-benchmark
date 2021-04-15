import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 
import services.hyperparameter as hyper

iterasi=10

for i in range(0,iterasi):
    X,y=data.baca_data_bank()
    model=gbdt.xgboost_model()
    parameter=hyper.hyper_xgboost_gs()
    hasil=opt
    print("Iterasi ke-"+str(i+1)+"\n")
    data.simpan_hasil_hpc(hasil)

