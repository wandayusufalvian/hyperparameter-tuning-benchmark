import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper

print("experiment 23 : lightgbm + dataset 2 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model,
                            hyperparameter=hyper.hyper_lightgbm_rs(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="accuracy")

print("\n")



print("experiment 24 : lightgbm + dataset 2 + bayes search"+"\n")

experiments.exp_bayes_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model(),
                            hyperparameter=hyper.hyper_lightgbm_bo(),
                            seeds=[1,12,22,32,42,52,62,72,82,92],
                            eval_method="accuracy")

print("\n")