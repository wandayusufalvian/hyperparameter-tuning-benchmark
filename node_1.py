import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper


print("experiment 33 : xgboost + dataset 3 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_housing(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.xgboost_model_reg(),
                            hyperparameter=hyper.hyper_xgboost_rs(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="neg_mean_absolute_error")

print("\n")

print("experiment 34 : xgboost + dataset 3 + bayes search"+"\n")

experiments.exp_bayes_search(
                            dataset=reader_writer.baca_data_housing(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.xgboost_model_reg(),
                            hyperparameter=hyper.hyper_xgboost_bo(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="neg_mean_absolute_error")

print("\n")

# 43
print("experiment 43 : catboost + dataset 3 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_housing_catboost(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.catboost_model_2_reg(),
                            hyperparameter=hyper.hyper_catboost_rs(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="neg_mean_absolute_error")

print("\n")
