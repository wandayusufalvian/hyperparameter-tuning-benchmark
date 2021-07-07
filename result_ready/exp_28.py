# dataset 2 
# lightgbm + bayes search 

acc_best_index_each_seed=[153,157,5,150,64,56,131,170,76,52]

acc_seed_1=[0.28867197, 0.43939184, 0.44335768, 0.45537966, 0.45091823,
       0.44980153, 0.45723824, 0.44335791, 0.33998338, 0.45079416,
       0.45537989, 0.43443382, 0.44025955, 0.43889602, 0.31655677,
       0.45290203, 0.44236589, 0.44843899, 0.44385357, 0.44174608,
       0.4351777 , 0.4427391 , 0.45104145, 0.43046775, 0.38398437,
       0.45290057, 0.44013533, 0.44447415, 0.37543326, 0.44980222,
       0.44856368, 0.38993363, 0.28111049, 0.44422609, 0.4411272 ,
       0.35448728, 0.34940351, 0.44013549, 0.44658018, 0.4437295 ,
       0.40939386, 0.45228107, 0.44856314, 0.44496958, 0.44249065,
       0.44013579, 0.42538406, 0.440507  , 0.43765617, 0.43914324,
       0.44286248, 0.45451194, 0.44732291, 0.4092701 , 0.44794348,
       0.42513753, 0.44137464, 0.3205243 , 0.44286171, 0.44434993,
       0.44434923, 0.31866756, 0.43294653, 0.35101526, 0.44162286,
       0.425013  , 0.44125111, 0.44199522, 0.45178595, 0.43976374,
       0.4410032 , 0.44323399, 0.44286217, 0.44794402, 0.44831492,
       0.44063099, 0.4402594 , 0.43530131, 0.44075499, 0.4400118 ,
       0.4542661 , 0.45203339, 0.4362931 , 0.45215677, 0.38212656,
       0.45277766, 0.30924451, 0.45005074, 0.31445243, 0.44211937,
       0.44447346, 0.45290073, 0.38274498, 0.4412508 , 0.44496858,
       0.34853564, 0.45153804, 0.29003473, 0.45252975, 0.4445976 ,
       0.45252883, 0.44769665, 0.43827652, 0.45191009, 0.44856345,
       0.32808485, 0.44286171, 0.45463593, 0.44050769, 0.44546571,
       0.43703667, 0.45215792, 0.4413748 , 0.42030221, 0.45761045,
       0.44087898, 0.45141381, 0.44670463, 0.43654078, 0.44137549,
       0.43976381, 0.32262995, 0.4414991 , 0.45054518, 0.44781957,
       0.39935492, 0.44112696, 0.44236697, 0.43703682, 0.31754848,
       0.44149864, 0.43988773, 0.45166165, 0.44137472, 0.44100359,
       0.43852397, 0.43877226, 0.44001211, 0.4427381 , 0.34878439,
       0.43678784, 0.45240552, 0.42984702, 0.28842206, 0.45029796,
       0.44348175, 0.44905826, 0.45066917, 0.44608421, 0.4371602 ,
       0.44918272, 0.44484513, 0.45674304, 0.45823064, 0.44013564,
       0.34531421, 0.43542515, 0.43604435, 0.40654656, 0.44050708,
       0.45153689, 0.43492987, 0.44360566, 0.43183113, 0.44075537,
       0.38163036, 0.44360574, 0.43344211, 0.42637684, 0.44038324,
       0.42488893, 0.44125111, 0.40257688, 0.441871  , 0.4424905 ,
       0.38286913, 0.40592376, 0.4410029 , 0.30812935, 0.4406313 ,
       0.44608421, 0.44286278, 0.44496904, 0.44187092, 0.4421186 ,
       0.44682732, 0.44075491, 0.45166149, 0.45637144, 0.44286263,
       0.45240514, 0.44348244, 0.39724697, 0.44063099, 0.44992652,
       0.44100266, 0.4411275 , 0.44769565, 0.42352586, 0.34233923]
acc_seed_12=[0.43530154, 0.43604488, 0.44472129, 0.3917946 , 0.4498026 ,
       0.44546463, 0.44893519, 0.4194361 , 0.43914378, 0.42860802,
       0.44013556, 0.44843892, 0.33378682, 0.44273856, 0.44249088,
       0.4445976 , 0.4279869 , 0.44806748, 0.44125119, 0.40493235,
       0.45252967, 0.44509265, 0.44050723, 0.45265282, 0.43455697,
       0.44509281, 0.32201145, 0.44273918, 0.44199468, 0.41906274,
       0.35560022, 0.45091731, 0.444474  , 0.44187123, 0.44001165,
       0.44162294, 0.43183029, 0.44335799, 0.29635474, 0.43629348,
       0.42761623, 0.44137533, 0.45128982, 0.43579744, 0.3818778 ,
       0.43939161, 0.42823612, 0.44546455, 0.44100405, 0.44893627,
       0.41237068, 0.41336185, 0.40344567, 0.44100328, 0.45314787,
       0.44311031, 0.44001142, 0.44199453, 0.41732708, 0.44323476,
       0.4387718 , 0.39551307, 0.45252898, 0.4398875 , 0.44298632,
       0.44943063, 0.44025986, 0.43257517, 0.44298601, 0.45265297,
       0.38646384, 0.44075529, 0.43393816, 0.35783231, 0.43691222,
       0.44298647, 0.41088231, 0.44125073, 0.41038657, 0.43678892,
       0.43840051, 0.44112689, 0.45364637, 0.42575627, 0.45463547,
       0.44558824, 0.43294622, 0.4410029 , 0.4359212 , 0.4232768 ,
       0.43864865, 0.45339601, 0.34518945, 0.44695216, 0.40976776,
       0.44918372, 0.45017366, 0.4515375 , 0.44595991, 0.4416224 ,
       0.43889602, 0.43492926, 0.44459776, 0.37258174, 0.45004997,
       0.44843899, 0.43926838, 0.44273856, 0.43034353, 0.4437305 ,
       0.4514142 , 0.42253346, 0.44843876, 0.452157  , 0.45426449,
       0.43356688, 0.38832403, 0.4531494 , 0.44558855, 0.44025955,
       0.44472091, 0.43579767, 0.43852404, 0.45017327, 0.4417467 ,
       0.44980222, 0.45413981, 0.3657653 , 0.43963952, 0.44311077,
       0.41906174, 0.44224228, 0.44645573, 0.3967523 , 0.4380283 ,
       0.44211875, 0.45364445, 0.44125119, 0.44397741, 0.45190871,
       0.44422609, 0.45277712, 0.44013556, 0.44534171, 0.45166218,
       0.44843876, 0.4385242 , 0.45265382, 0.43691245, 0.44434908,
       0.43901986, 0.44025948, 0.43852466, 0.44707569, 0.44075537,
       0.44348175, 0.4421186 , 0.45810672, 0.43988704, 0.42873293,
       0.43046729, 0.45587478, 0.45364499, 0.44125111, 0.43195551,
       0.45141474, 0.44112689, 0.43654139, 0.44311038, 0.44373035,
       0.44397764, 0.44075506, 0.37555979, 0.45091808, 0.33477822,
       0.43703682, 0.4336891 , 0.4526529 , 0.38596794, 0.44831546,
       0.43492895, 0.2874302 , 0.45562756, 0.44459822, 0.44087921,
       0.44137441, 0.4515385 , 0.4434819 , 0.39043083, 0.34407413,
       0.45228161, 0.44732337, 0.4387718 , 0.35200482, 0.44769604,
       0.43554853, 0.4284841 , 0.43096319, 0.44038332, 0.43926708]
acc_seed_22=[0.439268  , 0.42216179, 0.44149879, 0.38807497, 0.44930709,
       0.45884937, 0.45525666, 0.4424905 , 0.4411272 , 0.45029873,
       0.37406727, 0.44112673, 0.41584024, 0.44472037, 0.45166211,
       0.43753279, 0.31680637, 0.44335852, 0.44286263, 0.44558862,
       0.45352146, 0.45104245, 0.44038362, 0.44063123, 0.41385759,
       0.43046768, 0.45364522, 0.30949542, 0.45116522, 0.44025932,
       0.43926846, 0.4442254 , 0.45525505, 0.43381355, 0.45773321,
       0.44038332, 0.45054587, 0.44496951, 0.45451209, 0.3712179 ,
       0.45228238, 0.44038362, 0.44199484, 0.43654039, 0.4396399 ,
       0.4525299 , 0.45376844, 0.44930633, 0.45017358, 0.4396399 ,
       0.45203324, 0.44249058, 0.44087921, 0.44199484, 0.43207881,
       0.38956242, 0.43914416, 0.43554976, 0.43517778, 0.45401658,
       0.37196155, 0.44013541, 0.4441014 , 0.44063153, 0.43988788,
       0.43790469, 0.4428624 , 0.45054541, 0.42377331, 0.35324643,
       0.44075491, 0.43753233, 0.44174654, 0.44001165, 0.44100282,
       0.44125088, 0.42674821, 0.43059044, 0.38138083, 0.44211867,
       0.42129484, 0.43629356, 0.43963975, 0.45352054, 0.45240599,
       0.45327332, 0.43666477, 0.45228199, 0.44620866, 0.4505461 ,
       0.31358471, 0.43926846, 0.45327224, 0.44310969, 0.39080381,
       0.33577024, 0.4536443 , 0.31668245, 0.45364507, 0.38856994,
       0.44075537, 0.30850125, 0.41348607, 0.45178564, 0.3313092 ,
       0.44868744, 0.35733619, 0.44075529, 0.45277796, 0.43059159,
       0.42612901, 0.4428624 , 0.45079393, 0.44744813, 0.43988788,
       0.44224221, 0.45104214, 0.44087898, 0.44769657, 0.45153765,
       0.44323476, 0.45661912, 0.44050739, 0.44025955, 0.43418561,
       0.44831531, 0.44137518, 0.4499269 , 0.42959888, 0.43790485,
       0.4426148 , 0.42600356, 0.44856268, 0.43877195, 0.44422532,
       0.45240437, 0.44323507, 0.34940351, 0.39960052, 0.43207781,
       0.43678892, 0.28111049, 0.44050708, 0.44781934, 0.45178564,
       0.43802823, 0.45401543, 0.41906374, 0.41683272, 0.38993478,
       0.45042303, 0.4511666 , 0.44893496, 0.44100266, 0.43604458,
       0.45699171, 0.45091777, 0.4538929 , 0.45352061, 0.37034941,
       0.4402594 , 0.44013579, 0.44087959, 0.42488909, 0.44137541,
       0.4384002 , 0.42179058, 0.35882441, 0.45339654, 0.4488112 ,
       0.44273833, 0.45884799, 0.44174654, 0.43716051, 0.44757158,
       0.44248988, 0.4377807 , 0.4201786 , 0.44385303, 0.44335814,
       0.43170706, 0.44905865, 0.44025986, 0.3730771 , 0.45389321,
       0.45191032, 0.44187107, 0.45240576, 0.44025925, 0.42476509,
       0.44918272, 0.44211837, 0.43071466, 0.43542546, 0.43232564,
       0.44174677, 0.4502985 , 0.30106554, 0.44843968, 0.35981427]
acc_seed_32=[0.44224221, 0.44149864, 0.43802792, 0.44831531, 0.45376891,
       0.30255291, 0.45066978, 0.44100336, 0.44744774, 0.44125157,
       0.45153589, 0.44137518, 0.43901986, 0.44695223, 0.44050716,
       0.44645619, 0.43964028, 0.38795052, 0.44236674, 0.43393778,
       0.44211891, 0.44174654, 0.43579667, 0.4391437 , 0.44174723,
       0.43877164, 0.44509404, 0.39848658, 0.43393824, 0.41088238,
       0.45067063, 0.44521726, 0.44211852, 0.42203826, 0.44087883,
       0.32721752, 0.44100343, 0.40604806, 0.45141474, 0.45228153,
       0.44348267, 0.36328583, 0.43530093, 0.44596114, 0.44422609,
       0.44868736, 0.43034338, 0.45810611, 0.42724464, 0.44038332,
       0.44434923, 0.42228563, 0.33415903, 0.4348058 , 0.44112704,
       0.34754677, 0.44434947, 0.37592862, 0.44100366, 0.44298609,
       0.4349301 , 0.45067063, 0.45463593, 0.44596075, 0.44025948,
       0.4349298 , 0.30961918, 0.44794341, 0.44149917, 0.419807  ,
       0.43951575, 0.44273764, 0.45327255, 0.44992752, 0.4519094 ,
       0.4402594 , 0.43616926, 0.42910376, 0.44211944, 0.4479444 ,
       0.43592151, 0.42067495, 0.44224228, 0.4183194 , 0.35473327,
       0.42203795, 0.4291043 , 0.44472175, 0.44211821, 0.45290088,
       0.41484892, 0.43592151, 0.44484513, 0.44372996, 0.44311008,
       0.44472198, 0.44298539, 0.43926777, 0.45327224, 0.45042287,
       0.44063123, 0.441871  , 0.43703544, 0.43778001, 0.42067411,
       0.43703667, 0.28111049, 0.44608421, 0.45562795, 0.45599985,
       0.45277827, 0.43877218, 0.44149879, 0.40356959, 0.44323422,
       0.33502844, 0.45153896, 0.44422609, 0.44273887, 0.40617143,
       0.43988765, 0.43740888, 0.4205505 , 0.45042264, 0.31097941,
       0.41559287, 0.44955508, 0.44137549, 0.4511666 , 0.39724819,
       0.44038378, 0.2866867 , 0.43232687, 0.44397803, 0.44484521,
       0.42241023, 0.44149902, 0.45711524, 0.43567375, 0.38113208,
       0.44695185, 0.31420099, 0.4362931 , 0.45488338, 0.43988788,
       0.39501694, 0.44087883, 0.45178733, 0.3511391 , 0.44050754,
       0.45897406, 0.45289988, 0.45178472, 0.45513106, 0.4416227 ,
       0.43430968, 0.43889633, 0.43840005, 0.45314902, 0.43542553,
       0.43542515, 0.28111049, 0.43877187, 0.43641671, 0.44298593,
       0.32449076, 0.43963959, 0.43592181, 0.37158896, 0.44162278,
       0.4403837 , 0.44001134, 0.44819208, 0.43517686, 0.43430906,
       0.45079424, 0.43294584, 0.44162255, 0.4413748 , 0.30354416,
       0.33552202, 0.44199484, 0.45352077, 0.44769542, 0.44769527,
       0.44063099, 0.36985452, 0.44335891, 0.44137518, 0.30540305,
       0.42699642, 0.43852427, 0.44211937, 0.45253021, 0.44112666,
       0.39253725, 0.44149864, 0.45624684, 0.44311   , 0.45327347]
acc_seed_42=[0.44571231, 0.44174677, 0.45178564, 0.44658064, 0.44372996,
       0.35783162, 0.36352975, 0.45463562, 0.44707638, 0.45463516,
       0.41869184, 0.45215777, 0.44174662, 0.43554914, 0.4289803 ,
       0.45153765, 0.45042272, 0.38026621, 0.44025963, 0.44608429,
       0.44633258, 0.44013572, 0.44112727, 0.43914408, 0.44286163,
       0.45190925, 0.43728481, 0.43864765, 0.42327718, 0.43716051,
       0.43604388, 0.44620751, 0.44286271, 0.44434931, 0.44249019,
       0.44298693, 0.45066963, 0.4356739 , 0.44385388, 0.45426472,
       0.4546357 , 0.44100343, 0.45587478, 0.45290173, 0.34444718,
       0.43914354, 0.44434985, 0.3135841 , 0.4437295 , 0.44273818,
       0.45302595, 0.44149917, 0.28111049, 0.42910376, 0.43914385,
       0.43059128, 0.43889617, 0.44236643, 0.43753264, 0.43245064,
       0.37196078, 0.4444733 , 0.38001692, 0.44893519, 0.45723878,
       0.43579697, 0.44249119, 0.44187015, 0.4476955 , 0.44273802,
       0.42278213, 0.45203378, 0.32833338, 0.37419272, 0.42340187,
       0.44187061, 0.31581358, 0.44360589, 0.45240568, 0.38125684,
       0.450919  , 0.4410029 , 0.43716097, 0.44323438, 0.448315  ,
       0.4406313 , 0.4496783 , 0.44397756, 0.44273879, 0.30887307,
       0.44558862, 0.43505379, 0.4501742 , 0.44149917, 0.44943116,
       0.35510509, 0.41373183, 0.43579744, 0.43480588, 0.43939184,
       0.44211891, 0.44112735, 0.44187084, 0.35386793, 0.45104284,
       0.33230091, 0.44286271, 0.44695131, 0.44360574, 0.4522823 ,
       0.44893427, 0.44980168, 0.45240576, 0.44918272, 0.43951575,
       0.28111049, 0.44298639, 0.4532727 , 0.43604511, 0.44323461,
       0.44174708, 0.4416227 , 0.43963944, 0.38547098, 0.44075552,
       0.45104237, 0.44967838, 0.44162363, 0.44298678, 0.41249344,
       0.44001211, 0.41881545, 0.28557146, 0.40294947, 0.44546471,
       0.39749649, 0.4410029 , 0.44112673, 0.44930656, 0.4411275 ,
       0.42848456, 0.31915869, 0.44100305, 0.44025979, 0.44806655,
       0.4552552 , 0.43121086, 0.45463662, 0.42265745, 0.452159  ,
       0.44050739, 0.45190902, 0.37555472, 0.45401658, 0.40307416,
       0.40964753, 0.44335814, 0.45178618, 0.42823642, 0.38460257,
       0.41583993, 0.45252921, 0.45302472, 0.37543395, 0.431458  ,
       0.43406239, 0.44893512, 0.44174747, 0.44596198, 0.44013579,
       0.42352563, 0.444474  , 0.45476046, 0.44187061, 0.30490739,
       0.45277743, 0.44496974, 0.44695239, 0.37233283, 0.44187084,
       0.44323468, 0.44013541, 0.31519562, 0.43963982, 0.42464187,
       0.40530456, 0.4382756 , 0.4406313 , 0.44496974, 0.43753172,
       0.41807111, 0.450795  , 0.450795  , 0.42203703, 0.45451232,
       0.45153827, 0.38262137, 0.35708828, 0.44261411, 0.42897977]
acc_seed_52=[0.41807134, 0.4504218 , 0.4499259 , 0.44038355, 0.44075599,
       0.45215792, 0.43740865, 0.44087875, 0.45265228, 0.44571246,
       0.39687629, 0.42414359, 0.44509304, 0.4529008 , 0.4322028 ,
       0.44187031, 0.43455789, 0.44794341, 0.44211852, 0.43455812,
       0.45141343, 0.34023228, 0.33527404, 0.44013564, 0.44100313,
       0.439268  , 0.44881151, 0.43740888, 0.44025925, 0.4403837 ,
       0.45277873, 0.45265389, 0.43133532, 0.4382759 , 0.34047973,
       0.41745214, 0.44236612, 0.45215754, 0.44087906, 0.44075499,
       0.32189014, 0.441871  , 0.43368956, 0.45091808, 0.44050716,
       0.44224267, 0.38931282, 0.43864796, 0.44311054, 0.44187061,
       0.44087921, 0.45091823, 0.44571262, 0.45376906, 0.44781926,
       0.44149864, 0.4594698 , 0.3217663 , 0.43951606, 0.34630723,
       0.35002101, 0.43901963, 0.44472137, 0.43629356, 0.38076187,
       0.40270064, 0.45475962, 0.44211829, 0.43108649, 0.44509319,
       0.44967853, 0.44868705, 0.4413751 , 0.42042635, 0.44187092,
       0.45413965, 0.45537981, 0.44298632, 0.44410217, 0.44658018,
       0.34927967, 0.45215754, 0.43195451, 0.45079401, 0.44112689,
       0.35572605, 0.44707615, 0.43393793, 0.44645573, 0.45339608,
       0.3614261 , 0.44112712, 0.44187046, 0.43294576, 0.30490485,
       0.45252844, 0.42340133, 0.44348236, 0.44174662, 0.43914393,
       0.44273787, 0.44546509, 0.35931984, 0.45203401, 0.41162703,
       0.45004997, 0.44335729, 0.30540266, 0.42092163, 0.44249004,
       0.44187069, 0.44868675, 0.44236681, 0.44087875, 0.44162286,
       0.4412505 , 0.4362931 , 0.30961972, 0.44013556, 0.44819062,
       0.44050723, 0.43133532, 0.45327309, 0.44087982, 0.44187084,
       0.44323415, 0.4527775 , 0.34952911, 0.44558832, 0.45129028,
       0.44050777, 0.44137472, 0.44484482, 0.44930579, 0.29908289,
       0.45178618, 0.44087898, 0.45674319, 0.43852389, 0.39637948,
       0.44211921, 0.45451255, 0.45215769, 0.45364468, 0.44224351,
       0.44187061, 0.43691237, 0.45253029, 0.41869046, 0.45302495,
       0.43319436, 0.44224228, 0.44050723, 0.43678892, 0.3991057 ,
       0.42699619, 0.34878531, 0.38596933, 0.44608467, 0.44100282,
       0.45265313, 0.4485636 , 0.45203401, 0.43914347, 0.44125073,
       0.44050731, 0.44112689, 0.43951614, 0.44075529, 0.45215654,
       0.44224236, 0.41856639, 0.44521657, 0.4448449 , 0.44261395,
       0.44769542, 0.43418576, 0.43592151, 0.44298632, 0.45129036,
       0.45265236, 0.44112681, 0.44980245, 0.44075537, 0.45302649,
       0.44732321, 0.37443902, 0.45190909, 0.45426457, 0.43108687,
       0.44125134, 0.43740919, 0.44608475, 0.43716089, 0.45153765,
       0.43691229, 0.45265197, 0.44546494, 0.43542569, 0.43406177]
acc_seed_62=[0.45736185, 0.44273779, 0.31184974, 0.44806732, 0.45190979,
       0.45302457, 0.43517763, 0.4538929 , 0.44311077, 0.45079354,
       0.44645619, 0.44199491, 0.44323422, 0.43901971, 0.44385419,
       0.44236674, 0.45228084, 0.31011392, 0.45166188, 0.43579767,
       0.45339777, 0.44025917, 0.44224213, 0.31271776, 0.45438979,
       0.43108603, 0.35274885, 0.45352084, 0.43852435, 0.44732413,
       0.44311038, 0.44335898, 0.45240437, 0.41993092, 0.44236643,
       0.43207873, 0.44087875, 0.45339685, 0.41088177, 0.45066948,
       0.42761715, 0.36762473, 0.44509235, 0.43418584, 0.43852404,
       0.44794356, 0.36142648, 0.39228949, 0.44075545, 0.44075514,
       0.44757166, 0.4396399 , 0.44100343, 0.45389213, 0.44335814,
       0.4490598 , 0.4406313 , 0.33787727, 0.36192046, 0.35398877,
       0.42984764, 0.4267479 , 0.447075  , 0.44100397, 0.4448449 ,
       0.35312113, 0.44447353, 0.45302595, 0.44856383, 0.44360666,
       0.44682909, 0.44236704, 0.4072866 , 0.44224313, 0.41546926,
       0.45389305, 0.43406108, 0.44534018, 0.30428765, 0.44063092,
       0.40220582, 0.32349744, 0.44236658, 0.45017358, 0.40406387,
       0.44608444, 0.42984787, 0.45302495, 0.44323445, 0.43096326,
       0.43939161, 0.28111049, 0.39761794, 0.44905911, 0.45339716,
       0.44360559, 0.44955462, 0.43976335, 0.43839997, 0.33576832,
       0.40555101, 0.34791828, 0.33403458, 0.4411272 , 0.4437295 ,
       0.44459783, 0.45302518, 0.44162278, 0.45079408, 0.45413965,
       0.33490237, 0.44843907, 0.45005013, 0.44125096, 0.44794348,
       0.4369126 , 0.4505461 , 0.38696027, 0.44100305, 0.44459791,
       0.44075499, 0.43616934, 0.31333665, 0.4164602 , 0.45079424,
       0.43356665, 0.44360605, 0.45661866, 0.43232649, 0.35225549,
       0.33428125, 0.4597187 , 0.42774015, 0.43889617, 0.43282307,
       0.4489345 , 0.44980107, 0.35361818, 0.45699133, 0.44831584,
       0.44199461, 0.44137472, 0.28111049, 0.44025948, 0.35436382,
       0.42414552, 0.44087906, 0.41385721, 0.43344258, 0.45265336,
       0.3567163 , 0.45252913, 0.44261418, 0.44868759, 0.44719876,
       0.34853694, 0.43840036, 0.30503084, 0.44125157, 0.28111049,
       0.45488468, 0.4428624 , 0.44236689, 0.38906714, 0.44472198,
       0.44211829, 0.44063153, 0.38014229, 0.45290134, 0.45166226,
       0.36799648, 0.45290234, 0.45017366, 0.43815245, 0.45054656,
       0.44943109, 0.44311062, 0.45785836, 0.44385403, 0.3761753 ,
       0.33502698, 0.45104261, 0.36340883, 0.33862115, 0.44583668,
       0.36625958, 0.44001172, 0.44595999, 0.43505333, 0.32188738,
       0.43406185, 0.43753264, 0.4410029 , 0.43988781, 0.44360597,
       0.44249073, 0.43716089, 0.39638017, 0.44682832, 0.44100366]
acc_seed_72=[0.44335852, 0.45599754, 0.4542638 , 0.42910468, 0.43839936,
       0.43889633, 0.44162255, 0.42426905, 0.45228176, 0.42203749,
       0.44348129, 0.43083881, 0.43914385, 0.44013564, 0.45153773,
       0.44534133, 0.44211944, 0.44149871, 0.38522506, 0.30726163,
       0.44174631, 0.43530185, 0.45066994, 0.44125142, 0.4271198 ,
       0.44881082, 0.44992713, 0.43319413, 0.43914347, 0.44769542,
       0.43691245, 0.43988804, 0.4447206 , 0.38745593, 0.44038324,
       0.435425  , 0.44137541, 0.43468212, 0.4360455 , 0.44980253,
       0.44174647, 0.31941083, 0.30738593, 0.42178958, 0.43505402,
       0.45488215, 0.44125065, 0.43121125, 0.45166188, 0.44707592,
       0.44261449, 0.44509311, 0.38633716, 0.29722277, 0.42786422,
       0.44434993, 0.44782064, 0.44038339, 0.44286294, 0.42811182,
       0.45215677, 0.44273856, 0.45178549, 0.44211852, 0.43914385,
       0.44521695, 0.44719976, 0.43976397, 0.43728442, 0.39464466,
       0.3516343 , 0.43530147, 0.44199514, 0.44943047, 0.43046722,
       0.44038347, 0.45091915, 0.44261403, 0.45228122, 0.4246411 ,
       0.44385357, 0.44075568, 0.45277735, 0.44558824, 0.45203324,
       0.4408789 , 0.44125088, 0.36056215, 0.44967884, 0.44323415,
       0.44645649, 0.39526439, 0.41670827, 0.40889912, 0.43071512,
       0.45104314, 0.44187084, 0.36105297, 0.41435225, 0.3288275 ,
       0.44571308, 0.44819154, 0.44063138, 0.37059479, 0.44050769,
       0.45166042, 0.44385419, 0.45129074, 0.44211829, 0.42687151,
       0.44385419, 0.45215746, 0.44261465, 0.35547638, 0.43852335,
       0.29598392, 0.45067047, 0.45624607, 0.45277789, 0.42513715,
       0.3138317 , 0.44249111, 0.43988773, 0.45166157, 0.43108733,
       0.45351946, 0.41360876, 0.43827598, 0.44050762, 0.33130828,
       0.45215746, 0.37047464, 0.43480596, 0.44893404, 0.45550373,
       0.45203324, 0.4412508 , 0.45228268, 0.45079393, 0.35175875,
       0.45215746, 0.44682863, 0.44050716, 0.41869123, 0.4579825 ,
       0.41831833, 0.44571254, 0.4260041 , 0.4355493 , 0.44087921,
       0.44137518, 0.44050716, 0.45153773, 0.447075  , 0.44967823,
       0.44075476, 0.3425873 , 0.44335829, 0.44930633, 0.44013541,
       0.44323461, 0.44249019, 0.41100561, 0.45476008, 0.43369002,
       0.43839974, 0.44682824, 0.4499259 , 0.38708419, 0.43443398,
       0.4607088 , 0.43369025, 0.43604442, 0.45066978, 0.44211875,
       0.32424339, 0.4407546 , 0.44670348, 0.43641724, 0.29623036,
       0.3729528 , 0.45116614, 0.45314848, 0.43592158, 0.43009593,
       0.44955485, 0.43170614, 0.44249011, 0.43926761, 0.42240993,
       0.44893558, 0.45971786, 0.28148224, 0.45265374, 0.42129399,
       0.44236704, 0.45463693, 0.43678838, 0.38534829, 0.42414521]
acc_seed_82=[0.44372943, 0.32089628, 0.44112766, 0.44571223, 0.45426572,
       0.44608352, 0.32771587, 0.38026613, 0.44620843, 0.43480503,
       0.38299251, 0.40877582, 0.44112689, 0.39340435, 0.42910345,
       0.4530261 , 0.44149963, 0.43939153, 0.43505425, 0.44955516,
       0.45252883, 0.44719953, 0.43877203, 0.45215754, 0.43716074,
       0.45091784, 0.43976374, 0.28111049, 0.45376944, 0.43294591,
       0.44348236, 0.45104168, 0.445712  , 0.43939145, 0.44434962,
       0.44236628, 0.44137549, 0.45005051, 0.40294955, 0.45401612,
       0.42042697, 0.45327278, 0.44075537, 0.40604806, 0.44187046,
       0.4500499 , 0.36861628, 0.44955493, 0.30193341, 0.44100305,
       0.45376837, 0.44100305, 0.44075568, 0.4282355 , 0.45054633,
       0.42674767, 0.4475712 , 0.30961542, 0.4429867 , 0.40257865,
       0.44360559, 0.42885608, 0.45091792, 0.45017427, 0.45686649,
       0.44620812, 0.44137526, 0.36056069, 0.29400034, 0.36799486,
       0.44893481, 0.44174708, 0.42699581, 0.44025948, 0.43579659,
       0.43827598, 0.4636847 , 0.4417467 , 0.45711524, 0.44174731,
       0.45079431, 0.44583607, 0.43901978, 0.44075483, 0.43244933,
       0.44335852, 0.45017481, 0.43145677, 0.34642976, 0.45240368,
       0.43877164, 0.46058519, 0.45290057, 0.44087906, 0.34618362,
       0.44249027, 0.45042257, 0.43988758, 0.45562779, 0.45128882,
       0.44149871, 0.29870653, 0.38063496, 0.35907124, 0.44286263,
       0.45277773, 0.44843976, 0.43939184, 0.31804376, 0.45141412,
       0.43951575, 0.44100336, 0.44125088, 0.44125126, 0.35200866,
       0.44311015, 0.40517942, 0.43542446, 0.44261426, 0.43926815,
       0.44757181, 0.45438871, 0.44360551, 0.45042218, 0.43926738,
       0.43108679, 0.44596045, 0.4410029 , 0.44645596, 0.38757747,
       0.42848449, 0.42327818, 0.44050739, 0.44249119, 0.44125134,
       0.44137487, 0.44397795, 0.45141358, 0.44311023, 0.42377377,
       0.45934657, 0.44831508, 0.44335845, 0.45054694, 0.34147351,
       0.44261411, 0.44174677, 0.36390633, 0.43592158, 0.44831454,
       0.45525551, 0.43926777, 0.44063115, 0.40034701, 0.45699171,
       0.39402439, 0.42625316, 0.43232672, 0.42067495, 0.44013556,
       0.44620836, 0.45203362, 0.45277696, 0.44335814, 0.44162263,
       0.32746543, 0.44087898, 0.4416227 , 0.40047039, 0.44955508,
       0.40480828, 0.45500929, 0.42823596, 0.43406192, 0.44348244,
       0.42364962, 0.45104291, 0.45401628, 0.44125096, 0.42798752,
       0.44125103, 0.28581929, 0.44162286, 0.44112712, 0.33254812,
       0.43368956, 0.4488102 , 0.44360612, 0.43418645, 0.43046729,
       0.41732823, 0.44930694, 0.43455782, 0.44496889, 0.44112696,
       0.44806709, 0.44112758, 0.45141397, 0.4447216 , 0.43418553]
acc_seed_92=[0.43964013, 0.44224259, 0.44348236, 0.44063184, 0.3759257 ,
       0.45562787, 0.44224228, 0.44385334, 0.42823619, 0.44001172,
       0.44125057, 0.45178656, 0.34308357, 0.44236666, 0.44112696,
       0.44149856, 0.44348229, 0.44211875, 0.45240552, 0.40679132,
       0.43579782, 0.44769642, 0.43926777, 0.41596531, 0.4522813 ,
       0.45352046, 0.4468287 , 0.44472144, 0.45290096, 0.44211867,
       0.32746558, 0.45265336, 0.34531137, 0.4335668 , 0.44125119,
       0.45451155, 0.44149933, 0.4545124 , 0.34766899, 0.39910724,
       0.44496974, 0.4325744 , 0.43753318, 0.42848318, 0.32783741,
       0.44955377, 0.44013564, 0.43294584, 0.44261434, 0.44311015,
       0.4524056 , 0.45785805, 0.45798304, 0.28073875, 0.4159647 ,
       0.44187046, 0.45066955, 0.43939138, 0.4527772 , 0.42129399,
       0.4410029 , 0.44187138, 0.44199491, 0.4286084 , 0.42240947,
       0.42451749, 0.44025917, 0.43926808, 0.28111049, 0.44063138,
       0.45079385, 0.44397703, 0.44645665, 0.43815222, 0.44063076,
       0.44534079, 0.44112696, 0.44236628, 0.44534087, 0.44385357,
       0.4526532 , 0.4364174 , 0.44025979, 0.28135833, 0.39749226,
       0.38261953, 0.44534133, 0.4511656 , 0.44583707, 0.38175658,
       0.45537973, 0.43765617, 0.44695308, 0.44645719, 0.44410133,
       0.45352023, 0.44162309, 0.44298639, 0.44930686, 0.44311038,
       0.4371602 , 0.44199453, 0.4416224 , 0.43864819, 0.43034353,
       0.44943101, 0.45153727, 0.44149887, 0.44087913, 0.44806763,
       0.324365  , 0.33874529, 0.44001142, 0.44360628, 0.45624876,
       0.44757181, 0.43059121, 0.43802884, 0.43220181, 0.44298632,
       0.43542546, 0.4360445 , 0.43691283, 0.43678892, 0.43616826,
       0.44881089, 0.4460859 , 0.4355493 , 0.4433576 , 0.43443398,
       0.43530208, 0.40828046, 0.4406313 , 0.45389359, 0.45104214,
       0.40989075, 0.44930694, 0.44125111, 0.44199399, 0.44050677,
       0.36006195, 0.44868736, 0.44335799, 0.45067063, 0.44087913,
       0.45091831, 0.35820368, 0.43641671, 0.43827613, 0.4540175 ,
       0.42550797, 0.29015634, 0.43926731, 0.44373012, 0.44992567,
       0.45091792, 0.44174662, 0.43604557, 0.44360636, 0.36650749,
       0.44881135, 0.37084654, 0.36663309, 0.43542507, 0.44335822,
       0.45029857, 0.42650022, 0.42798821, 0.39328166, 0.44806748,
       0.4538929 , 0.44100405, 0.39675261, 0.44112743, 0.45277727,
       0.39501748, 0.43195482, 0.44050777, 0.45265313, 0.44050746,
       0.42439335, 0.40468406, 0.3207716 , 0.4402594 , 0.45277604,
       0.44707538, 0.45290073, 0.44249004, 0.4449692 , 0.43220265,
       0.44744797, 0.4522813 , 0.40381696, 0.45649497, 0.44633258,
       0.43963967, 0.44211967, 0.43889617, 0.44881128, 0.45129013]

acc_all_each_seed=[acc_seed_1,acc_seed_12,acc_seed_22,acc_seed_32,acc_seed_42,acc_seed_52,acc_seed_62,acc_seed_72,acc_seed_82,acc_seed_92]

acc_best_score_mean=0.458
acc_best_score_std=0.001
exec_time_mean=1503



