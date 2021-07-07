# dataset 2 
# xgboost + random search 

acc_best_index_each_seed=[18,129,131,119,59,48,93,157,67,7]

acc_seed_1=[0.51524632, 0.51598974, 0.52987184, 0.53259975, 0.51227165,
       0.52268296, 0.52987145, 0.49665343, 0.53173195, 0.52974769,
       0.50198265, 0.50285022, 0.53210286, 0.53445833, 0.5241701 ,
       0.52404634, 0.52231013, 0.51437945, 0.53978871, 0.5286313 ,
       0.52875613, 0.50347118, 0.52268365, 0.53086362, 0.52850876,
       0.53396175, 0.52107205, 0.49219408, 0.51598659, 0.53767991,
       0.53173073, 0.52987168, 0.50607418, 0.50359479, 0.53111145,
       0.50495763, 0.50768493, 0.51722951, 0.51747657, 0.53235176,
       0.51437899, 0.52181409, 0.5203284 , 0.5149998 , 0.53334286,
       0.52702185, 0.51586459, 0.51437876, 0.52503789, 0.52677333,
       0.53346647, 0.49690118, 0.50136315, 0.51797293, 0.53210286,
       0.52826024, 0.5385487 , 0.51946115, 0.53297127, 0.51760026,
       0.52330323, 0.53148197, 0.51970975, 0.53259913, 0.52838362,
       0.50966935, 0.51909078, 0.52590561, 0.51760095, 0.53421065,
       0.50148714, 0.52925226, 0.53396167, 0.52863222, 0.52925149,
       0.52045117, 0.501363  , 0.52379758, 0.5014866 , 0.52565678,
       0.52094714, 0.53197879, 0.50706681, 0.50471041, 0.53173134,
       0.53631699, 0.50409053, 0.50619763, 0.53061449, 0.53036758,
       0.49169611, 0.52887959, 0.51289154, 0.48822416, 0.51350904,
       0.52950086, 0.5164864 , 0.53458133, 0.53383775, 0.53483092,
       0.52826047, 0.53173057, 0.53222746, 0.51264417, 0.5211942 ,
       0.53111161, 0.49566318, 0.52825986, 0.50719019, 0.51685853,
       0.49132382, 0.50954627, 0.5220643 , 0.49987486, 0.4926869 ,
       0.52243528, 0.52466699, 0.52503751, 0.52850792, 0.48835154,
       0.53755692, 0.529624  , 0.53445879, 0.50855334, 0.51983228,
       0.5161145 , 0.53222677, 0.50000023, 0.4966532 , 0.53247499,
       0.49652967, 0.49070501, 0.51834598, 0.49925581, 0.5330951 ,
       0.52776389, 0.49913228, 0.52838393, 0.52553225, 0.5330958 ,
       0.51289246, 0.51735258, 0.52590477, 0.49454717, 0.53185595,
       0.51735466, 0.52479121, 0.51710659, 0.52156841, 0.49467024,
       0.52950001, 0.53359077, 0.53421058, 0.51834614, 0.53396236,
       0.53433411, 0.52900274, 0.52590523, 0.50123985, 0.53160765,
       0.5292518 , 0.51512233, 0.53012021, 0.51809615, 0.53135829,
       0.52231121, 0.52702139, 0.53086332, 0.50012445, 0.53222777,
       0.53160535, 0.52231182, 0.52825939, 0.53346631, 0.53768122,
       0.53210316, 0.52813517, 0.52342768, 0.52181532, 0.53371537,
       0.51350927, 0.52875583, 0.51537062, 0.52950001, 0.51958445,
       0.53359008, 0.49950495, 0.53445825, 0.50458611, 0.50557882,
       0.53272228, 0.48921687, 0.52094906, 0.51375703, 0.50941921,
       0.53148328, 0.52652419, 0.53743277, 0.53780444, 0.52875644]
acc_seed_12=[0.48971207, 0.51871542, 0.51735281, 0.51326275, 0.46876724,
       0.50904977, 0.51685776, 0.52255904, 0.52875729, 0.52416995,
       0.53160735, 0.53098746, 0.5069412 , 0.51363611, 0.5199575 ,
       0.52726945, 0.51623619, 0.52503843, 0.52181555, 0.5158659 ,
       0.53668905, 0.52900304, 0.53197948, 0.53483039, 0.49653005,
       0.49504253, 0.50706519, 0.51214804, 0.53074009, 0.50619832,
       0.53433403, 0.50818266, 0.52640143, 0.49417512, 0.53135859,
       0.51214827, 0.53247453, 0.53346754, 0.49442433, 0.46529752,
       0.53507822, 0.52416925, 0.53073925, 0.49603416, 0.52057631,
       0.52008003, 0.51921332, 0.52987099, 0.53532651, 0.52689801,
       0.53086424, 0.53235084, 0.50830558, 0.53049226, 0.51636256,
       0.51623903, 0.52429378, 0.52987114, 0.53359084, 0.5345824 ,
       0.52603029, 0.53483008, 0.53544843, 0.49553773, 0.50979303,
       0.53582133, 0.53693703, 0.53259883, 0.53309457, 0.51809838,
       0.53780429, 0.53185495, 0.52169201, 0.5099161 , 0.52280749,
       0.52082314, 0.47856104, 0.50632201, 0.52206299, 0.47843459,
       0.53569718, 0.53160742, 0.51574168, 0.53693688, 0.52293079,
       0.51041368, 0.53743431, 0.53978824, 0.52268242, 0.52429409,
       0.50694143, 0.5140064 , 0.50842973, 0.53123614, 0.516733  ,
       0.5091733 , 0.51350904, 0.48661426, 0.51127994, 0.49380245,
       0.53086347, 0.53421035, 0.52949924, 0.52838508, 0.53049095,
       0.5075617 , 0.53297011, 0.50719072, 0.53594463, 0.50768616,
       0.53123629, 0.5298703 , 0.5002486 , 0.5313589 , 0.50309935,
       0.53185464, 0.52503736, 0.50322358, 0.52726922, 0.51908855,
       0.52206468, 0.52925118, 0.51908955, 0.53321933, 0.53383806,
       0.52367397, 0.50842957, 0.50371901, 0.53520244, 0.54028421,
       0.49070394, 0.49467247, 0.52082407, 0.51351127, 0.52416956,
       0.52082468, 0.48190354, 0.52255935, 0.52875575, 0.52516142,
       0.50929552, 0.52987207, 0.53359015, 0.53098685, 0.52255942,
       0.53495338, 0.52156618, 0.49392859, 0.52070077, 0.53334255,
       0.53668951, 0.51301599, 0.45761221, 0.53284543, 0.52181532,
       0.50966935, 0.51475104, 0.50619825, 0.50247924, 0.52578131,
       0.52962331, 0.52528511, 0.52069869, 0.53805266, 0.4971514 ,
       0.52540903, 0.52590484, 0.48946386, 0.5354502 , 0.53346754,
       0.48314362, 0.53036696, 0.52714515, 0.52590538, 0.52801249,
       0.5200808 , 0.48041793, 0.53210209, 0.50607264, 0.505083  ,
       0.52119558, 0.52875682, 0.51710606, 0.53086255, 0.53098785,
       0.49467001, 0.49256306, 0.52578193, 0.52949932, 0.52987207,
       0.4384022 , 0.53074071, 0.5350776 , 0.52639974, 0.52094867,
       0.5200798 , 0.50185858, 0.53755669, 0.52181447, 0.52627805]
acc_seed_22=[0.5199552 , 0.4618265 , 0.50855311, 0.51400663, 0.5073131 ,
       0.51549292, 0.5353259 , 0.53160789, 0.52776458, 0.49491869,
       0.53309434, 0.51772525, 0.53396275, 0.50880132, 0.5342105 ,
       0.49405174, 0.50384385, 0.52937579, 0.52107159, 0.51413008,
       0.53743216, 0.50347171, 0.53197971, 0.52293186, 0.53049141,
       0.527144  , 0.52355137, 0.52119466, 0.53532474, 0.53656651,
       0.51028885, 0.52379858, 0.53173057, 0.53148405, 0.52020449,
       0.5292518 , 0.53247437, 0.50371985, 0.52020249, 0.50644554,
       0.52801302, 0.49913566, 0.53222708, 0.51115687, 0.52739275,
       0.5080576 , 0.50260315, 0.52528519, 0.5333434 , 0.53111145,
       0.53916705, 0.53854893, 0.52726799, 0.51871612, 0.51252056,
       0.52280764, 0.52342591, 0.5282607 , 0.53111115, 0.52206376,
       0.52826009, 0.52987237, 0.53743162, 0.53123552, 0.53693695,
       0.527145  , 0.53408566, 0.5081812 , 0.52528588, 0.53420981,
       0.52280772, 0.52268303, 0.53297027, 0.52280656, 0.52070053,
       0.52788826, 0.53507683, 0.51078428, 0.51227112, 0.53706141,
       0.52937679, 0.50297421, 0.49392698, 0.53235053, 0.53148251,
       0.53284658, 0.53718502, 0.52702124, 0.53148343, 0.52974731,
       0.52937594, 0.51363611, 0.50545529, 0.53148312, 0.50024745,
       0.49913267, 0.53817673, 0.52949955, 0.51375749, 0.53953957,
       0.46170251, 0.51685669, 0.51834468, 0.51722851, 0.53197802,
       0.5260306 , 0.50991702, 0.53235123, 0.52875606, 0.52962362,
       0.53755661, 0.49467016, 0.52714408, 0.51834575, 0.52987153,
       0.50818259, 0.53197879, 0.53470601, 0.50880009, 0.53098823,
       0.53284773, 0.4882247 , 0.51165207, 0.50012399, 0.50309805,
       0.51536885, 0.5059508 , 0.53557196, 0.5316075 , 0.51995727,
       0.5359457 , 0.54139915, 0.52479052, 0.53334309, 0.50285022,
       0.53693695, 0.49876007, 0.52429471, 0.50830343, 0.50979272,
       0.52057616, 0.46566904, 0.52776412, 0.51871834, 0.53904383,
       0.51326198, 0.50347171, 0.52280718, 0.51004148, 0.53284681,
       0.52231021, 0.50458749, 0.53272336, 0.51388263, 0.53173219,
       0.53235069, 0.52726853, 0.51276755, 0.53730955, 0.52801302,
       0.52602876, 0.51648463, 0.52875583, 0.52912796, 0.53321864,
       0.4967775 , 0.52429386, 0.50979395, 0.52417025, 0.4877295 ,
       0.46951012, 0.53321802, 0.50359525, 0.52801249, 0.53929212,
       0.51537008, 0.52528649, 0.52429309, 0.53135921, 0.53061556,
       0.52478937, 0.5294997 , 0.53259983, 0.52627713, 0.53941596,
       0.52974869, 0.53681365, 0.50929698, 0.53086431, 0.52888005,
       0.53483046, 0.52553279, 0.53334278, 0.49851293, 0.53222777,
       0.51697953, 0.52231244, 0.532227  , 0.5350773 , 0.50297613]
acc_seed_32=[0.53458194, 0.53495407, 0.51970783, 0.53582225, 0.51066075,
       0.49479469, 0.52788849, 0.50780999, 0.52342668, 0.53086293,
       0.5345824 , 0.52280718, 0.50681836, 0.53396251, 0.53495407,
       0.50223048, 0.53706171, 0.53098808, 0.4484413 , 0.52813602,
       0.43579836, 0.53396167, 0.52900412, 0.51004194, 0.53408612,
       0.52553379, 0.49119999, 0.51908786, 0.53334217, 0.5200798 ,
       0.53284704, 0.49045588, 0.50706642, 0.5157426 , 0.48971284,
       0.52652473, 0.53210286, 0.51338858, 0.52094775, 0.52416841,
       0.51983359, 0.50198311, 0.52516242, 0.53284704, 0.51710606,
       0.51983336, 0.52293086, 0.48599583, 0.50756278, 0.52268273,
       0.49838863, 0.5290035 , 0.53569895, 0.53284735, 0.46827096,
       0.52206322, 0.5190884 , 0.53346578, 0.52974815, 0.53296942,
       0.51574283, 0.53135936, 0.52553402, 0.53359092, 0.50743809,
       0.53197909, 0.52603037, 0.52107282, 0.53160704, 0.52082353,
       0.52131881, 0.52850823, 0.51772487, 0.48450892, 0.5225595 ,
       0.52082422, 0.51834368, 0.51115633, 0.50569989, 0.50533084,
       0.50508331, 0.53024381, 0.49355415, 0.52788803, 0.52726961,
       0.50954505, 0.51660817, 0.4861189 , 0.5220633 , 0.52999575,
       0.5178484 , 0.52144503, 0.53867177, 0.5347067 , 0.52590669,
       0.50669422, 0.53656605, 0.51896387, 0.51066059, 0.5210719 ,
       0.50756163, 0.49727385, 0.5324756 , 0.53507753, 0.4954142 ,
       0.5177264 , 0.53247545, 0.53148343, 0.48376143, 0.52813609,
       0.50595034, 0.50471149, 0.53396205, 0.53123568, 0.49715055,
       0.53061602, 0.52367374, 0.51958429, 0.53532551, 0.54040698,
       0.5270217 , 0.5221853 , 0.50210664, 0.51413154, 0.53644175,
       0.50644623, 0.52404764, 0.52788688, 0.53086339, 0.51487365,
       0.50012315, 0.5338386 , 0.5396641 , 0.53185472, 0.50285168,
       0.52429348, 0.52380004, 0.4983881 , 0.52243466, 0.52875475,
       0.51970836, 0.53111084, 0.53371468, 0.47434629, 0.53297119,
       0.51983021, 0.52565701, 0.52181378, 0.51041084, 0.52578193,
       0.51846798, 0.53445802, 0.50768585, 0.52875613, 0.50656999,
       0.50768554, 0.52912873, 0.50000023, 0.53173134, 0.50756239,
       0.52181516, 0.52008096, 0.53768099, 0.51524625, 0.51363503,
       0.51450298, 0.53594578, 0.53421012, 0.53743246, 0.53309395,
       0.53371368, 0.5344591 , 0.50917291, 0.53297057, 0.53346724,
       0.50619863, 0.52838377, 0.509049  , 0.50334634, 0.53879646,
       0.51946007, 0.53210316, 0.5337143 , 0.52280772, 0.52813579,
       0.53247576, 0.51177668, 0.53334294, 0.52664849, 0.53272197,
       0.51388187, 0.53941673, 0.51152655, 0.52900496, 0.53111245,
       0.5194603 , 0.50780984, 0.51797454, 0.5353262 , 0.52801172]
acc_seed_42=[0.50012376, 0.51140432, 0.51462667, 0.47583297, 0.52553417,
       0.53086378, 0.51921347, 0.50421452, 0.53086347, 0.53284712,
       0.51623865, 0.54028314, 0.52813663, 0.50483609, 0.52342745,
       0.51660993, 0.53073986, 0.51698083, 0.52565716, 0.51661016,
       0.51970952, 0.5060741 , 0.52355121, 0.5168583 , 0.49603585,
       0.52751675, 0.51165069, 0.50470949, 0.52714531, 0.50656992,
       0.52194108, 0.48785395, 0.51512364, 0.53383875, 0.5208233 ,
       0.532227  , 0.53830087, 0.52243474, 0.4384022 , 0.53036719,
       0.5277655 , 0.51487419, 0.49219085, 0.53470647, 0.47855804,
       0.52131881, 0.52801279, 0.53470732, 0.53297104, 0.51995689,
       0.53408628, 0.50880001, 0.50037152, 0.52454254, 0.50371855,
       0.51586513, 0.48698624, 0.49578548, 0.53780436, 0.54053166,
       0.52987176, 0.48896866, 0.49256383, 0.51140417, 0.54040759,
       0.52974777, 0.52069869, 0.51450206, 0.52764143, 0.50644654,
       0.51970798, 0.46120685, 0.51549454, 0.50582704, 0.51883972,
       0.52788726, 0.52999544, 0.53160727, 0.5237995 , 0.53061571,
       0.52925164, 0.53185472, 0.53197902, 0.53891953, 0.5325989 ,
       0.53148282, 0.53507768, 0.52540887, 0.52503889, 0.51376056,
       0.5195833 , 0.46219863, 0.53706056, 0.53594524, 0.53259829,
       0.5156163 , 0.51760049, 0.53358954, 0.52677271, 0.53755669,
       0.52317793, 0.51078474, 0.52503643, 0.53259936, 0.53681319,
       0.49479592, 0.53185395, 0.49690149, 0.4981408 , 0.52677325,
       0.50842857, 0.52293025, 0.51437822, 0.51760026, 0.52590561,
       0.53792905, 0.50495924, 0.52069892, 0.50123923, 0.51586698,
       0.52590446, 0.51958422, 0.51487511, 0.52640051, 0.52082322,
       0.51537131, 0.5365649 , 0.51512233, 0.50694228, 0.53123568,
       0.48859683, 0.51574237, 0.49826441, 0.514254  , 0.52714415,
       0.52193785, 0.5380525 , 0.53148328, 0.5322267 , 0.53507753,
       0.52751728, 0.5299956 , 0.53445864, 0.52082414, 0.51090827,
       0.45934765, 0.51784901, 0.49950403, 0.51375964, 0.52565763,
       0.52664903, 0.48586946, 0.53346662, 0.52206353, 0.50879986,
       0.53135898, 0.53259829, 0.53049103, 0.52689786, 0.52962377,
       0.52887874, 0.53197925, 0.53445918, 0.52342668, 0.53879638,
       0.52528695, 0.52082483, 0.520576  , 0.52751621, 0.52479022,
       0.53197848, 0.50495678, 0.52553402, 0.51611289, 0.52987138,
       0.53073955, 0.5146279 , 0.50471126, 0.52937571, 0.52627651,
       0.52813617, 0.52863122, 0.49913244, 0.53916897, 0.50780784,
       0.53656636, 0.51363373, 0.5237962 , 0.53483016, 0.53148351,
       0.51549508, 0.53854863, 0.52107159, 0.53272236, 0.51983228,
       0.52057501, 0.5304918 , 0.49268774, 0.50855364, 0.50731418]
acc_seed_52=[0.52875537, 0.53470486, 0.51698283, 0.50892531, 0.53346562,
       0.53197879, 0.53892014, 0.52702177, 0.52912665, 0.52925203,
       0.5089267 , 0.52912842, 0.5112801 , 0.53978755, 0.52925203,
       0.53582171, 0.52466576, 0.53148305, 0.52144234, 0.50272784,
       0.51462751, 0.53024351, 0.4621984 , 0.52528603, 0.53061648,
       0.50706589, 0.53135959, 0.53359046, 0.49851232, 0.51239557,
       0.52615229, 0.5304908 , 0.53173103, 0.50409099, 0.51326283,
       0.52887759, 0.52999537, 0.52776511, 0.52702085, 0.51599105,
       0.52776527, 0.49467024, 0.52838462, 0.52603045, 0.53445833,
       0.52429386, 0.52739391, 0.51735212, 0.53978786, 0.52466576,
       0.49405066, 0.50867525, 0.48525111, 0.5204524 , 0.51599012,
       0.5137601 , 0.51983105, 0.53197833, 0.50396769, 0.53247522,
       0.53321856, 0.53718509, 0.5324753 , 0.53148266, 0.52540895,
       0.48537633, 0.53978702, 0.52863153, 0.53173149, 0.52949894,
       0.53904421, 0.53036804, 0.53111138, 0.53445733, 0.52776496,
       0.49058133, 0.51586713, 0.53483008, 0.51004125, 0.53830033,
       0.50595103, 0.50694243, 0.51586613, 0.53185495, 0.53061533,
       0.5261526 , 0.52516258, 0.53185503, 0.52826093, 0.53123476,
       0.53594617, 0.48785318, 0.53433518, 0.52751667, 0.5344591 ,
       0.51797324, 0.53197917, 0.52181493, 0.53334248, 0.52045178,
       0.52925264, 0.52156687, 0.52342691, 0.50756239, 0.51586736,
       0.5298723 , 0.52565862, 0.53012074, 0.53049226, 0.52156841,
       0.53693657, 0.52156864, 0.50830566, 0.53297019, 0.50942098,
       0.52293063, 0.51326421, 0.49962695, 0.52751759, 0.5083052 ,
       0.52962285, 0.52119581, 0.53458056, 0.53148312, 0.50855357,
       0.53544989, 0.53396305, 0.53185541, 0.52850838, 0.49702548,
       0.53235222, 0.53408728, 0.53359031, 0.51636241, 0.51338812,
       0.52292971, 0.5254088 , 0.52416956, 0.5337143 , 0.52987138,
       0.52912796, 0.53185541, 0.53284689, 0.53644214, 0.52912834,
       0.51437945, 0.52181593, 0.49541297, 0.52950032, 0.53557403,
       0.52441778, 0.52788911, 0.5120239 , 0.53197956, 0.48785295,
       0.50793191, 0.53916805, 0.53098692, 0.53383929, 0.50793383,
       0.51809715, 0.51375864, 0.5338386 , 0.52082476, 0.48624182,
       0.51809661, 0.53284658, 0.52454131, 0.53049157, 0.53433449,
       0.53383952, 0.52850738, 0.50607548, 0.52925157, 0.48760512,
       0.53371491, 0.49789289, 0.52999529, 0.52788849, 0.51090758,
       0.53743254, 0.53631768, 0.53148405, 0.51375818, 0.53098762,
       0.50669414, 0.51599012, 0.50632293, 0.48351706, 0.5114044 ,
       0.53557434, 0.52045301, 0.52454254, 0.53160719, 0.52640104,
       0.50161114, 0.50867741, 0.52565739, 0.52008011, 0.51512141]
acc_seed_62=[0.5286316 , 0.50062004, 0.50966858, 0.53631799, 0.5173522 ,
       0.53334286, 0.53049088, 0.52007888, 0.48376366, 0.53197909,
       0.52416925, 0.52763974, 0.52987153, 0.50284937, 0.53321856,
       0.52776389, 0.52479068, 0.53706125, 0.53693764, 0.51822176,
       0.51896379, 0.52193908, 0.50384139, 0.53061425, 0.53024374,
       0.53383998, 0.5318548 , 0.52937525, 0.51673469, 0.5177261 ,
       0.48797764, 0.5081819 , 0.53135936, 0.51400601, 0.51785155,
       0.51338697, 0.50533053, 0.52900496, 0.51772548, 0.5329708 ,
       0.52726792, 0.53854863, 0.53421065, 0.50669414, 0.50483433,
       0.53148412, 0.51636256, 0.53173073, 0.52169094, 0.53235138,
       0.51908978, 0.50049536, 0.52379866, 0.51276662, 0.50409352,
       0.52937579, 0.52974777, 0.53421012, 0.53272359, 0.49702479,
       0.50681875, 0.5176021 , 0.51611458, 0.53359154, 0.5174778 ,
       0.50371916, 0.53520167, 0.53470632, 0.53197848, 0.49467009,
       0.51574283, 0.51636095, 0.52454177, 0.5169816 , 0.49181979,
       0.521939  , 0.50892424, 0.51834368, 0.52689601, 0.50942105,
       0.51661201, 0.53210247, 0.52045339, 0.50421429, 0.53532559,
       0.49950372, 0.51351212, 0.51090981, 0.51834606, 0.53507829,
       0.51152662, 0.52144472, 0.52850869, 0.5397874 , 0.53334332,
       0.51016462, 0.53222746, 0.53966471, 0.51413016, 0.49516598,
       0.52540926, 0.52255812, 0.51760241, 0.53507776, 0.52131904,
       0.52454185, 0.50446312, 0.5006195 , 0.53383921, 0.53458179,
       0.53631737, 0.53606862, 0.52342545, 0.5214428 , 0.50557813,
       0.52949963, 0.52008111, 0.53743262, 0.53247391, 0.53235138,
       0.50433874, 0.53619361, 0.51648648, 0.50223125, 0.52999575,
       0.53668905, 0.49987585, 0.52850784, 0.51487481, 0.5192114 ,
       0.52553371, 0.52962354, 0.53830064, 0.50037113, 0.52565647,
       0.52801195, 0.51722867, 0.52999591, 0.53172996, 0.51066036,
       0.49120022, 0.53235077, 0.49913228, 0.53284697, 0.5291265 ,
       0.51326283, 0.52739245, 0.51673224, 0.5199552 , 0.51165207,
       0.52342568, 0.5349533 , 0.49132452, 0.53445833, 0.53247583,
       0.51388379, 0.538672  , 0.52255935, 0.52466545, 0.52144257,
       0.5110318 , 0.51549546, 0.50223071, 0.5324756 , 0.53135967,
       0.51363565, 0.53297142, 0.53681296, 0.51859366, 0.51896364,
       0.53024343, 0.52206345, 0.5356968 , 0.532227  , 0.52714415,
       0.53334363, 0.53011882, 0.50681828, 0.53346639, 0.52590507,
       0.53371461, 0.50508216, 0.51797193, 0.51549477, 0.50731402,
       0.53619277, 0.5288802 , 0.51115603, 0.53272251, 0.5168573 ,
       0.51723005, 0.52243559, 0.53135821, 0.52094752, 0.52317977,
       0.49752069, 0.50942075, 0.53631745, 0.5361943 , 0.51958422]
acc_seed_72=[0.52466653, 0.52404572, 0.53792866, 0.5174778 , 0.51623918,
       0.5197079 , 0.53173134, 0.52255881, 0.53321848, 0.52255827,
       0.51958476, 0.52082276, 0.51090935, 0.53371361, 0.49479562,
       0.52801256, 0.5266491 , 0.51487542, 0.53073932, 0.51896333,
       0.53681219, 0.52404557, 0.53049203, 0.51561661, 0.53594517,
       0.53098631, 0.51797393, 0.52057516, 0.51921278, 0.48611744,
       0.51995712, 0.51809815, 0.53111084, 0.48314385, 0.52317847,
       0.53482977, 0.501362  , 0.53483   , 0.51933569, 0.52937679,
       0.52962347, 0.49343101, 0.53681281, 0.4877295 , 0.51995558,
       0.52069908, 0.52020433, 0.51946168, 0.50495909, 0.51933777,
       0.52540895, 0.51648663, 0.53024366, 0.53098746, 0.53383852,
       0.52218622, 0.52689794, 0.53309472, 0.52900435, 0.51685922,
       0.52243574, 0.53718563, 0.53309395, 0.50223087, 0.52354937,
       0.53916675, 0.53346716, 0.50272599, 0.52069977, 0.53210378,
       0.52999598, 0.53210209, 0.524542  , 0.51698229, 0.51586705,
       0.53235107, 0.52491329, 0.52788819, 0.46120723, 0.50681928,
       0.5216924 , 0.52714454, 0.52912796, 0.50619878, 0.52764074,
       0.52652573, 0.53817719, 0.51970806, 0.52776458, 0.52664864,
       0.49107768, 0.52850892, 0.51400394, 0.52317854, 0.5353249 ,
       0.52664972, 0.50842873, 0.50508339, 0.53879661, 0.51351181,
       0.52194038, 0.52441762, 0.52950024, 0.53086339, 0.50768646,
       0.51784886, 0.49392629, 0.5344591 , 0.52268234, 0.48958839,
       0.53334271, 0.53098731, 0.52367367, 0.5334667 , 0.53049203,
       0.51487542, 0.52962278, 0.46727987, 0.5233043 , 0.53768114,
       0.53247499, 0.50656999, 0.53495415, 0.53297134, 0.53185433,
       0.52503712, 0.51413108, 0.5375563 , 0.52863268, 0.52801149,
       0.50408991, 0.52987107, 0.49876023, 0.53383929, 0.51623803,
       0.48611952, 0.48066584, 0.53644198, 0.49603362, 0.49491876,
       0.52826093, 0.52602876, 0.52578139, 0.49863631, 0.52367551,
       0.50793199, 0.51797201, 0.50656838, 0.53284689, 0.50842942,
       0.51772533, 0.53197963, 0.50371793, 0.50099133, 0.49417435,
       0.50285129, 0.53780513, 0.54152299, 0.52565786, 0.52615429,
       0.52045232, 0.50297551, 0.53768114, 0.52987199, 0.49739646,
       0.53383921, 0.51103219, 0.5319794 , 0.51896379, 0.51660924,
       0.51152831, 0.43827829, 0.50309905, 0.51078336, 0.49863478,
       0.46070888, 0.50446243, 0.51537124, 0.53321771, 0.5016106 ,
       0.53830103, 0.52974838, 0.51921262, 0.53433349, 0.52677279,
       0.51784932, 0.53532613, 0.5322257 , 0.51537154, 0.52863206,
       0.52169163, 0.5096675 , 0.53197986, 0.52987207, 0.48711061,
       0.50210711, 0.47124679, 0.51561853, 0.52528611, 0.52925326]
acc_seed_82=[0.51995504, 0.51127971, 0.52429348, 0.50582527, 0.49665351,
       0.53197871, 0.53197863, 0.5192117 , 0.53842479, 0.53594494,
       0.53247545, 0.518098  , 0.5089257 , 0.51475143, 0.51462782,
       0.50570181, 0.49937988, 0.53185572, 0.5328462 , 0.53197902,
       0.52082307, 0.52342607, 0.53359084, 0.52764128, 0.50384292,
       0.51338728, 0.51177607, 0.49937927, 0.53991239, 0.51016324,
       0.51772418, 0.50731356, 0.53173165, 0.5314832 , 0.50433767,
       0.53111138, 0.52255873, 0.52045309, 0.52441854, 0.53086339,
       0.52416948, 0.53656482, 0.53111153, 0.53272228, 0.52863191,
       0.50520623, 0.51871642, 0.53197848, 0.53445841, 0.53259875,
       0.51004094, 0.52255912, 0.46864271, 0.52801149, 0.53297127,
       0.50458657, 0.53160666, 0.50768569, 0.50495809, 0.52429501,
       0.48760443, 0.53346662, 0.52962324, 0.53482985, 0.52379827,
       0.5171049 , 0.52912765, 0.54065558, 0.50309743, 0.53309334,
       0.50247862, 0.53297027, 0.51822184, 0.5042139 , 0.51549431,
       0.53495269, 0.53296981, 0.5322267 , 0.52429471, 0.49789213,
       0.53197863, 0.52479091, 0.53358915, 0.50718996, 0.52293094,
       0.51140417, 0.52751613, 0.50322288, 0.48946463, 0.50099071,
       0.51933738, 0.49975232, 0.5354502 , 0.5149988 , 0.52032671,
       0.49764491, 0.50619932, 0.53991139, 0.45376967, 0.48723268,
       0.53135975, 0.52677287, 0.52801133, 0.5325996 , 0.50607418,
       0.53606916, 0.50049489, 0.53371499, 0.47583251, 0.53433403,
       0.51078443, 0.5268974 , 0.50396738, 0.48946516, 0.52776419,
       0.50731272, 0.52156879, 0.53173234, 0.5259053 , 0.53222723,
       0.53830033, 0.5355745 , 0.51797231, 0.52652673, 0.49541404,
       0.51970898, 0.53321841, 0.5313589 , 0.48797787, 0.52962377,
       0.53148435, 0.51883965, 0.50533091, 0.53433426, 0.53557319,
       0.49504191, 0.53309457, 0.51648594, 0.53743308, 0.52701978,
       0.49541381, 0.53148343, 0.52937663, 0.51859205, 0.50396676,
       0.52454185, 0.52714561, 0.52528572, 0.51326167, 0.48772996,
       0.53086393, 0.52962485, 0.50446342, 0.517974  , 0.48500351,
       0.50247908, 0.4947937 , 0.53098731, 0.51388379, 0.48128404,
       0.5301189 , 0.53049218, 0.51834406, 0.52739337, 0.53135982,
       0.51227127, 0.48550132, 0.50433905, 0.51859235, 0.53966333,
       0.51437845, 0.51549438, 0.52764174, 0.52330338, 0.51115603,
       0.53544981, 0.53272336, 0.53321871, 0.5151234 , 0.5384244 ,
       0.52751713, 0.5301199 , 0.53309495, 0.49231515, 0.52677225,
       0.53297104, 0.54003469, 0.53495446, 0.52032694, 0.46294167,
       0.53272297, 0.52330269, 0.52231059, 0.49306018, 0.53173111,
       0.53123514, 0.47583267, 0.52404618, 0.50508277, 0.52764043]
acc_seed_92=[0.50817967, 0.52342783, 0.51784947, 0.52888043, 0.49851278,
       0.48673856, 0.51797347, 0.53966418, 0.53359031, 0.51363473,
       0.52900443, 0.52937533, 0.4615779 , 0.53297019, 0.52169194,
       0.48996014, 0.48004496, 0.52875583, 0.52937679, 0.51313945,
       0.52317885, 0.52032702, 0.49950418, 0.53346585, 0.50384308,
       0.53693619, 0.49491792, 0.51512325, 0.52677294, 0.53259921,
       0.52466607, 0.51846782, 0.51896449, 0.51103188, 0.52218722,
       0.50694128, 0.52268226, 0.51970798, 0.51896479, 0.52578085,
       0.53111284, 0.49764537, 0.53346555, 0.52739198, 0.52317739,
       0.52838631, 0.51524671, 0.52466492, 0.52094714, 0.5074384 ,
       0.4952899 , 0.48252389, 0.53135906, 0.51475212, 0.53086301,
       0.53185564, 0.53346624, 0.49603362, 0.52094637, 0.52764089,
       0.52268334, 0.53359069, 0.52578193, 0.52280641, 0.53123391,
       0.52169178, 0.51561907, 0.51933746, 0.53359046, 0.53259921,
       0.52206422, 0.52578116, 0.53681296, 0.52900289, 0.52491428,
       0.53210293, 0.51474943, 0.52664895, 0.43555037, 0.5215681 ,
       0.53954026, 0.53321871, 0.52218583, 0.51437807, 0.53073886,
       0.53768045, 0.50421552, 0.49677704, 0.52900327, 0.4925636 ,
       0.53842494, 0.51933738, 0.5280111 , 0.52206215, 0.52801233,
       0.53557311, 0.46083479, 0.52838462, 0.52417041, 0.51685823,
       0.52801172, 0.53693688, 0.52875552, 0.51103165, 0.53358977,
       0.52726738, 0.52218668, 0.52391919, 0.52404503, 0.51301446,
       0.51809631, 0.52677348, 0.497397  , 0.53297127, 0.5281354 ,
       0.50793284, 0.5241711 , 0.53569726, 0.53036642, 0.53854801,
       0.48921726, 0.52801225, 0.53135975, 0.51958376, 0.53111107,
       0.51983289, 0.53507768, 0.53334224, 0.52689655, 0.53346608,
       0.52999614, 0.5246653 , 0.52925164, 0.52479091, 0.51698114,
       0.51660963, 0.53792759, 0.51636241, 0.49677696, 0.53867124,
       0.53197963, 0.4970251 , 0.51921385, 0.52404657, 0.52429455,
       0.51599135, 0.51537108, 0.52107105, 0.52516142, 0.53780536,
       0.53098662, 0.50297459, 0.51078328, 0.52553425, 0.5035944 ,
       0.53160704, 0.53135929, 0.53173096, 0.53544989, 0.53173195,
       0.53433434, 0.50309874, 0.52503712, 0.52925333, 0.53086309,
       0.53049141, 0.51251887, 0.51400755, 0.53545027, 0.53693726,
       0.53297165, 0.53197886, 0.50867472, 0.53408589, 0.52342622,
       0.5390446 , 0.51685861, 0.53222716, 0.53507876, 0.51611412,
       0.51623788, 0.51388394, 0.53371384, 0.5318548 , 0.51524717,
       0.53086355, 0.53792728, 0.50781099, 0.51413008, 0.53235153,
       0.51797239, 0.4964056 , 0.52999521, 0.53321818, 0.53036773,
       0.52826085, 0.52850861, 0.50186004, 0.53482969, 0.51252079]

acc_all_each_seed=[acc_seed_1,acc_seed_12,acc_seed_22,acc_seed_32,acc_seed_42,acc_seed_52,acc_seed_62,acc_seed_72,acc_seed_82,acc_seed_92]

acc_best_score_mean=0.539
acc_best_score_std=0.0007
exec_time_mean=2450
