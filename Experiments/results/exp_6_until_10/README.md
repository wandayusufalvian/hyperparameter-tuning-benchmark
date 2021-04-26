experiment_7_1, experiment_8_1, experiment_9_1 =>> hyperparameter : hyper_lightgbm_rs dan hyper_lightgbm_bo

experiment_7_2, experiment_8_2 dan experiment_9_2 =>> hyperparameter : hyper_lightgbm_rs_2 dan hyper_lightgbm_bo_2
hyper_lightgbm_bo_2 sama saja dengan hyper_lightgbm_bo, bedanya hanya di stopping point. meskipun pada akhirnya 
ketika menggunakan hyper_lightgbm_bo_2 iterasinya tetap 200, kecuali yg random seed 22. oleh karena itu akan di run ulang yg random seed-nya 22 

experiment_9_3 = run ulang yg random seed nya 22 karena pada experiment sebelumnya iterasi tidak sama