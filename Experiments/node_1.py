import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper


# exp 16  

print("experiment 16 : xgboost + dataset 2 + default hyperparameter"+"\n")

experiments.exp_default(
                        dataset=reader_writer.baca_data_cus_seg(),
                        mesin=0,
                        file_name="--",
                        iterasi=10,
                        model=gbdt.xgboost_model(),
                        eval_method="accuracy")

print("\n")

# exp 21 : 

print("experiment 21 : lightgbm + dataset 2 + default hyperparameter"+"\n")

experiments.exp_default(
                        dataset=reader_writer.baca_data_cus_seg(),
                        mesin=0,
                        file_name="--",
                        iterasi=10,
                        model=gbdt.lightgbm_model(),
                        eval_method="accuracy")

print("\n")

# exp 26 :

print("experiment 26 : catboost + dataset 2 + default hyperparameter"+"\n")

experiments.exp_default(
                        dataset=reader_writer.baca_data_cus_seg_catboost(),
                        mesin=0,
                        file_name="--",
                        iterasi=10,
                        model=gbdt.catboost_model_2(),
                        eval_method="accuracy")

print("\n")