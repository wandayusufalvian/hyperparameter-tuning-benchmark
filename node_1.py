import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper


print("experiment 27 : catboost + dataset 2 + grid search"+"\n")

experiments.exp_grid_search(
                            dataset=reader_writer.baca_data_cus_seg_catboost(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.catboost_model_2(),
                            hyperparameter=hyper.hyper_catboost_gs(),
                            eval_method="accuracy")

print("\n")

print("experiment 28 : catboost + dataset 2 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_cus_seg_catboost(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.catboost_model_2(),
                            hyperparameter=hyper.hyper_catboost_rs(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="accuracy")

print("\n")

