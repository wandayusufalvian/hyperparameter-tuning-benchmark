# dataset 1 
# xgboost + grid search

exec_time=1693
auc_best_score=0.523
auc_best_std=0.229

best_index=8

auc_score=[0.51291962, 0.49192787, 0.50740776, 0.49183634, 0.51555368,
       0.49183634, 0.52364505, 0.49183634, 0.5249099 , 0.49296397,
       0.46279833, 0.40782903, 0.47167245, 0.40721245, 0.47997846,
       0.40643293, 0.48858729, 0.41005432, 0.49538052, 0.41218307,
       0.45644519, 0.38564367, 0.46887621, 0.38985428, 0.47790059,
       0.39273583, 0.48862215, 0.3983782 , 0.49481413, 0.40465346,
       0.4555091 , 0.38322605, 0.4688794 , 0.38960134, 0.47789985,
       0.39249739, 0.48862215, 0.39839523, 0.49481413, 0.40461311,
       0.455481  , 0.38305682, 0.4688794 , 0.38954898, 0.47789985,
       0.39248884, 0.48862215, 0.3983936 , 0.49481413, 0.40461311,
       0.51291962, 0.49192787, 0.50740911, 0.49183634, 0.51555368,
       0.49183634, 0.52364505, 0.49183634, 0.5249099 , 0.49296397,
       0.46183645, 0.40807684, 0.47072221, 0.40701025, 0.47916823,
       0.40632158, 0.48844581, 0.40986926, 0.4945675 , 0.41200515,
       0.45546542, 0.38521961, 0.46924146, 0.38973033, 0.47722841,
       0.39244282, 0.48866535, 0.39835383, 0.49432482, 0.40445495,
       0.45535788, 0.38343946, 0.46922559, 0.38930483, 0.47726143,
       0.39238704, 0.48866535, 0.39816537, 0.49432482, 0.40431532,
       0.45534655, 0.38338145, 0.46922559, 0.38934958, 0.47726143,
       0.39228067, 0.48866535, 0.39816463, 0.49432482, 0.40431532,
       0.48595857, 0.4807511 , 0.48128647, 0.47770374, 0.49702616,
       0.4760716 , 0.48606988, 0.48094168, 0.49736501, 0.47933207,
       0.4449809 , 0.40844397, 0.44895077, 0.4032564 , 0.45426693,
       0.40312864, 0.45418956, 0.40358218, 0.46182311, 0.4049916 ,
       0.45992853, 0.41839088, 0.4493436 , 0.41659231, 0.44971076,
       0.41234066, 0.44853765, 0.40907566, 0.46580065, 0.40930666,
       0.44953238, 0.41953476, 0.46070228, 0.41372518, 0.45408398,
       0.41405059, 0.44853765, 0.41104928, 0.46580065, 0.41055888,
       0.45908712, 0.41997477, 0.46070228, 0.41576616, 0.45408398,
       0.41781236, 0.44853765, 0.41360001, 0.46580065, 0.40750579,
       0.49928867, 0.4801581 , 0.48515837, 0.47858684, 0.49705611,
       0.47609394, 0.48606988, 0.47845651, 0.49905697, 0.48057324,
       0.45903562, 0.41179361, 0.45605856, 0.40284741, 0.45087611,
       0.40614718, 0.46501509, 0.40594376, 0.45554679, 0.40773418,
       0.47047251, 0.41575153, 0.46365687, 0.41332523, 0.45269602,
       0.40853525, 0.4581891 , 0.40976078, 0.45725843, 0.40610791,
       0.45282456, 0.42160868, 0.45504202, 0.41594508, 0.45310814,
       0.41126857, 0.4581891 , 0.40998896, 0.45725843, 0.40644422,
       0.45848994, 0.41731358, 0.45504202, 0.42225889, 0.45310814,
       0.41361183, 0.4581891 , 0.41003394, 0.45725843, 0.40612056]