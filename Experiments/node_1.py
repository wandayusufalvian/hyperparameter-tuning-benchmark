import experiments 
import services.reader_writer as reader_writer
import services.models as gbdt
import services.hyperparameter as hyper

# xgboost + .... + .... 

experiments.exp_default(
                        dataset=reader_writer.baca_data_bank(),
                        mesin=1,
                        file_name="exp_6_lightgbm_dataset1_default",
                        iterasi=1,
                        model=gbdt.lightgbm_model())