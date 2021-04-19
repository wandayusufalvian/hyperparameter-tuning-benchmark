import services.data as data
import services.gbdt as gbdt
import services.optimize as opt 

iterasi=10

for i in range(0,iterasi):
    X,y=data.baca_data_bank()
    model=gbdt.lightgbm_model()
    hasil=opt.default_hyperparameter(X,y,model)
    print("Iterasi ke-"+str(i+1)+"\n")
    data.simpan_hasil_hpc(hasil)