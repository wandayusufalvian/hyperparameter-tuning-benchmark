import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper



print("experiment 31 : xgboost + dataset 3 + default hyperparameter"+"\n")

experiments.exp_default(
                        dataset=reader_writer.baca_data_housing(),
                        mesin=0,
                        file_name="--",
                        iterasi=10,
                        model=gbdt.xgboost_model_reg(),
                        eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 36 : lightgbm + dataset 3 + default hyperparameter"+"\n")

experiments.exp_default(
                        dataset=reader_writer.baca_data_housing(),
                        mesin=0,
                        file_name="--",
                        iterasi=10,
                        model=gbdt.lightgbm_model_reg(),
                        eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 41 : catboost + dataset 3 + default hyperparameter"+"\n")

experiments.exp_default(
                        dataset=reader_writer.baca_data_housing_catboost(),
                        mesin=0,
                        file_name="--",
                        iterasi=10,
                        model=gbdt.catboost_model_2_reg(),
                        eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 32 : xgboost + dataset 3 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_housing(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.xgboost_model_reg(),
                            hyperparameter=hyper.hyper_xgboost_gs(),
                            eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 37 :lightgbm + dataset 3 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_housing(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model_reg(),
                            hyperparameter=hyper.hyper_lightgbm_gs(),
                            eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 42 :catboost + dataset 3 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_housing_catboost(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.catboost_model_2_reg(),
                            hyperparameter=hyper.hyper_catboost_gs(),
                            eval_method="neg_mean_absolute_error")
print("\n")


# 38, 39 

print("experiment 38 : lightgbm + dataset 3 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_housing(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model_reg(),
                            hyperparameter=hyper.hyper_lightgbm_rs(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 39 : lightgbm + dataset 3 + bayes search"+"\n")

experiments.exp_bayes_search(
                            dataset=reader_writer.baca_data_housing(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model_reg(),
                            hyperparameter=hyper.hyper_lightgbm_bo(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="neg_mean_absolute_error")

print("\n")

# 44
print("experiment 44 : catboost + dataset 3 + bayes search"+"\n")

experiments.exp_bayes_search(
                            dataset=reader_writer.baca_data_housing_catboost(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.catboost_model_2_reg(),
                            hyperparameter=hyper.hyper_catboost_bo(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="neg_mean_absolute_error")

print("\n")
