import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper


# exp 17  

print("experiment 17 : xgboost + dataset 2 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.xgboost_model(),
                            hyperparameter=hyper.hyper_xgboost_gs(),
                            eval_method="accuracy")

print("\n")

# exp 22 : 

print("experiment 22 : lightgbm + dataset 2 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model(),
                            hyperparameter=hyper.hyper_lightgbm_gs(),
                            eval_method="accuracy")

print("\n")

# exp 27 :

print("experiment 27 : catboost + dataset 2 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.catboost_model_2(),
                            hyperparameter=hyper.hyper_catboost_gs(),
                            eval_method="accuracy")

print("\n")