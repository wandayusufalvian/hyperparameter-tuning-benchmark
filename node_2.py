import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper

print("experiment 23_1 : lightgbm + dataset 2 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model(),
                            hyperparameter=hyper.hyper_lightgbm_gs(),
                            seeds=[1,12,22,32,42],
                            eval_method="accuracy")

print("\n")

print("experiment 23_2 : lightgbm + dataset 2 + random search"+"\n")

experiments.exp_random_search(
                            dataset=reader_writer.baca_data_cus_seg(),
                            mesin=0,
                            file_name="--",
                            model=gbdt.lightgbm_model(),
                            hyperparameter=hyper.hyper_lightgbm_gs(),
                            seeds=[52,62,72,82,92],
                            eval_method="accuracy")

print("\n")