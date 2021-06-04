# xgboost + random search + dataset 1 

auc_best_index_each_seed=[94,176,170,73,113,147,43,36,52,190]

auc_seed_1=[0.46412394, 0.45562413, 0.39465992, 0.40529901, 0.40926188,
       0.40528072, 0.43146799, 0.50631172, 0.41025114, 0.40377248,
       0.48893456, 0.49057025, 0.42340612, 0.42698355, 0.49482367,
       0.4344392 , 0.43251201, 0.46489951, 0.44261875, 0.4171657 ,
       0.41277716, 0.4784654 , 0.40493805, 0.39727001, 0.47668941,
       0.44691955, 0.44899217, 0.41655179, 0.4337907 , 0.45571636,
       0.3983882 , 0.42483368, 0.48446246, 0.47967152, 0.41054207,
       0.41298274, 0.49497871, 0.44604945, 0.46996649, 0.40932593,
       0.46191468, 0.43643203, 0.41298534, 0.46284855, 0.40053901,
       0.4880796 , 0.39081004, 0.40558715, 0.39929802, 0.4348913 ,
       0.47458824, 0.40364973, 0.51462907, 0.44609378, 0.42348215,
       0.45556718, 0.4171902 , 0.45122204, 0.4251465 , 0.43805649,
       0.40086672, 0.45484503, 0.45460739, 0.41432925, 0.45953484,
       0.46720656, 0.41450641, 0.43129834, 0.46629109, 0.40763374,
       0.4932903 , 0.40969783, 0.45503684, 0.43613088, 0.42357617,
       0.4343775 , 0.49339331, 0.4189348 , 0.48896767, 0.42601395,
       0.43810112, 0.42618418, 0.46574366, 0.48180804, 0.41498385,
       0.44116103, 0.48833158, 0.48289385, 0.42973885, 0.3919847 ,
       0.51848122, 0.41831359, 0.41585704, 0.50507354, 0.53372976,
       0.41510113, 0.45452479, 0.41695139, 0.42911941, 0.43511681,
       0.43619719, 0.42772544, 0.41331245, 0.46687752, 0.43816262,
       0.4108231 , 0.42770052, 0.42197935, 0.47707014, 0.45560879,
       0.49923106, 0.46321989, 0.44994445, 0.43728087, 0.5100428 ,
       0.43904701, 0.41845324, 0.43019865, 0.39210141, 0.41890898,
       0.42389751, 0.39005748, 0.40720354, 0.47533115, 0.40263954,
       0.4632446 , 0.48285144, 0.50539018, 0.49795295, 0.45465335,
       0.49892222, 0.50313312, 0.4544743 , 0.49495348, 0.4077546 ,
       0.42409553, 0.50011082, 0.43052717, 0.43514318, 0.39566136,
       0.46477437, 0.40761209, 0.425289  , 0.40081826, 0.40518484,
       0.41365897, 0.38926088, 0.46149385, 0.49734136, 0.5006741 ,
       0.41968375, 0.40185925, 0.41768914, 0.45113176, 0.41351003,
       0.41341319, 0.42557161, 0.43379093, 0.49579337, 0.40918696,
       0.43229323, 0.39771792, 0.42060006, 0.43943821, 0.39156698,
       0.44770289, 0.48567967, 0.43146458, 0.42091153, 0.40945767,
       0.43899074, 0.44399018, 0.46333879, 0.41702393, 0.42937527,
       0.43445431, 0.42571917, 0.44569737, 0.43717808, 0.401938  ,
       0.45048466, 0.4306219 , 0.45874872, 0.40512011, 0.46685891,
       0.41310224, 0.50365958, 0.40485134, 0.48559032, 0.47849786,
       0.48653066, 0.50319823, 0.45275483, 0.49651099, 0.47891818,
       0.40546539, 0.42645968, 0.42361014, 0.42572028, 0.48497361]
auc_seed_12=[0.50256478, 0.40377674, 0.45023731, 0.46379072, 0.425311  ,
       0.48154288, 0.46889278, 0.45196074, 0.41369503, 0.43239658,
       0.43676589, 0.39753445, 0.506172  , 0.45988868, 0.45597478,
       0.44038267, 0.50186187, 0.46044595, 0.46332721, 0.44742644,
       0.42296688, 0.42792376, 0.42322405, 0.42640199, 0.49721303,
       0.49432274, 0.47970527, 0.47024354, 0.44285088, 0.47183957,
       0.40926381, 0.46631674, 0.43723529, 0.50063   , 0.42560847,
       0.48639757, 0.42897893, 0.40905173, 0.50325763, 0.5149843 ,
       0.43898718, 0.43180705, 0.43536694, 0.50720322, 0.44876723,
       0.45513267, 0.4510298 , 0.43016167, 0.40774923, 0.40619379,
       0.41152481, 0.39900735, 0.48243139, 0.45054521, 0.45969097,
       0.46305058, 0.43064249, 0.44339269, 0.41877146, 0.41065552,
       0.40692606, 0.41061854, 0.43432908, 0.50193401, 0.47113771,
       0.40153039, 0.42804021, 0.41689713, 0.40091471, 0.45339194,
       0.42115219, 0.41698078, 0.44351615, 0.46862429, 0.40220381,
       0.45107691, 0.43305758, 0.47798898, 0.43391217, 0.44044963,
       0.41549315, 0.40930295, 0.50596893, 0.44892256, 0.42771575,
       0.39216523, 0.43114888, 0.44402468, 0.40629375, 0.42136431,
       0.40629204, 0.45565887, 0.46844493, 0.40969402, 0.44485686,
       0.47755651, 0.46038958, 0.4250767 , 0.47094763, 0.49847894,
       0.40382692, 0.41091899, 0.42381546, 0.41149933, 0.43762227,
       0.4670735 , 0.41962844, 0.46817141, 0.40813128, 0.42984922,
       0.41271647, 0.42565877, 0.50096666, 0.45520113, 0.48308191,
       0.39307975, 0.42654452, 0.47864369, 0.43409185, 0.40157515,
       0.45202375, 0.42706479, 0.41417815, 0.40192817, 0.44291261,
       0.40232361, 0.47603156, 0.482616  , 0.45261399, 0.44372174,
       0.50429067, 0.4001802 , 0.45823109, 0.4566435 , 0.43174142,
       0.451414  , 0.56273645, 0.44735697, 0.4306532 , 0.49028456,
       0.5147451 , 0.40260464, 0.45453275, 0.42956517, 0.43364397,
       0.4183043 , 0.43497424, 0.41122831, 0.44421466, 0.421271  ,
       0.42099356, 0.46229507, 0.50131806, 0.43437353, 0.42942466,
       0.4648851 , 0.50427225, 0.48024219, 0.49852753, 0.42206656,
       0.41489521, 0.43207455, 0.424247  , 0.43602153, 0.42435985,
       0.43085114, 0.46338551, 0.51112487, 0.43260746, 0.39493479,
       0.50035798, 0.44813479, 0.44062414, 0.41406067, 0.43338689,
       0.44671607, 0.59024803, 0.42022566, 0.53648702, 0.47774615,
       0.43570677, 0.44813766, 0.45218827, 0.42753579, 0.40251532,
       0.50030544, 0.42683721, 0.41254691, 0.41788958, 0.42471496,
       0.52181371, 0.40084292, 0.43044462, 0.39111449, 0.45272634,
       0.43599696, 0.48973687, 0.42849869, 0.45594222, 0.45291279]
auc_seed_22=[0.4387763 , 0.52076065, 0.47219608, 0.38805449, 0.40700681,
       0.44141794, 0.4385979 , 0.4130322 , 0.42601261, 0.50327357,
       0.42336259, 0.48794122, 0.39924317, 0.46849019, 0.39737458,
       0.45590769, 0.42728112, 0.4356969 , 0.43644984, 0.45325224,
       0.42681506, 0.47603846, 0.41256565, 0.39939211, 0.42600688,
       0.38977662, 0.44311945, 0.43782122, 0.43092072, 0.46283318,
       0.4733772 , 0.4888855 , 0.40397794, 0.41014715, 0.461222  ,
       0.43845035, 0.42253085, 0.47781196, 0.43755953, 0.4728303 ,
       0.39840651, 0.41127154, 0.41804731, 0.46280589, 0.42526014,
       0.48189262, 0.47851319, 0.45459015, 0.41330672, 0.47944565,
       0.42038313, 0.4386848 , 0.42352359, 0.43639911, 0.47882172,
       0.45414107, 0.46583063, 0.46753855, 0.43119474, 0.45567646,
       0.40209688, 0.40853517, 0.43424096, 0.40225123, 0.45150392,
       0.41182851, 0.45846632, 0.42542694, 0.40572531, 0.41303047,
       0.45361887, 0.43577045, 0.40924624, 0.44349282, 0.44823516,
       0.4367206 , 0.48102616, 0.46140464, 0.47036427, 0.40954028,
       0.4456445 , 0.48480892, 0.50017213, 0.41798103, 0.43001226,
       0.42093683, 0.43144892, 0.44438558, 0.4200193 , 0.42311664,
       0.41522784, 0.46257639, 0.47598168, 0.40055301, 0.45746478,
       0.50257263, 0.42780369, 0.43571108, 0.45432612, 0.42297851,
       0.51969867, 0.5326797 , 0.45923775, 0.56108932, 0.42173502,
       0.40819408, 0.40508529, 0.41025647, 0.39848368, 0.41385905,
       0.44236951, 0.5055878 , 0.43165015, 0.46458493, 0.39184086,
       0.40304599, 0.42408493, 0.44628631, 0.4756351 , 0.46043148,
       0.40651472, 0.5005629 , 0.47303839, 0.49806185, 0.4913332 ,
       0.43601402, 0.48357702, 0.47035362, 0.410827  , 0.46074368,
       0.42425947, 0.42130085, 0.40188806, 0.45873965, 0.48858063,
       0.41688642, 0.5127438 , 0.43896275, 0.46106263, 0.48856689,
       0.48519155, 0.52176065, 0.41902642, 0.45335519, 0.45655186,
       0.49815429, 0.48092869, 0.43716593, 0.4818103 , 0.39604609,
       0.45689488, 0.48145889, 0.39914218, 0.41557487, 0.41991807,
       0.42786975, 0.42685605, 0.48195416, 0.4307796 , 0.46862717,
       0.42805289, 0.44099694, 0.43529463, 0.41553379, 0.46476383,
       0.49552852, 0.43347903, 0.46510433, 0.50900325, 0.50933325,
       0.60851512, 0.42069284, 0.47657562, 0.45484211, 0.42723769,
       0.45497406, 0.44843131, 0.40801512, 0.4307075 , 0.39906212,
       0.46257239, 0.41096464, 0.44799722, 0.48747709, 0.43746786,
       0.48134816, 0.43746998, 0.47298191, 0.41020709, 0.42010017,
       0.41493316, 0.43015852, 0.40683587, 0.50086846, 0.40695523,
       0.507474  , 0.4488567 , 0.43494355, 0.39872322, 0.41301524]
auc_seed_32=[0.42950183, 0.4327414 , 0.43992188, 0.46166282, 0.43276423,
       0.50175243, 0.486328  , 0.46855862, 0.43609597, 0.42873513,
       0.40992176, 0.54057377, 0.46404709, 0.43205133, 0.43645955,
       0.49342442, 0.4431645 , 0.40931882, 0.52651301, 0.42666178,
       0.52172886, 0.42291787, 0.44702533, 0.46320649, 0.42320901,
       0.39439661, 0.50165859, 0.43585749, 0.4170406 , 0.4743416 ,
       0.42308067, 0.49806425, 0.46539713, 0.46076915, 0.50954453,
       0.42669611, 0.43725007, 0.46597606, 0.42505158, 0.41207133,
       0.45224211, 0.4823072 , 0.43487049, 0.39807738, 0.45607783,
       0.44416376, 0.45359071, 0.43117137, 0.46795371, 0.43836786,
       0.4955981 , 0.42313153, 0.45921804, 0.4121966 , 0.52929528,
       0.43543945, 0.42169141, 0.41630624, 0.41513861, 0.42570564,
       0.48042451, 0.43848082, 0.43560505, 0.40383592, 0.48190898,
       0.39749957, 0.45080224, 0.44318879, 0.43518378, 0.49122985,
       0.43746023, 0.43508518, 0.45812207, 0.59343007, 0.44794007,
       0.45040144, 0.44045058, 0.41379802, 0.4868345 , 0.47551445,
       0.47527521, 0.43132354, 0.50661559, 0.42980177, 0.41365641,
       0.47624475, 0.44156269, 0.51269885, 0.4138552 , 0.39038089,
       0.43799156, 0.45130745, 0.41817964, 0.40545327, 0.44017668,
       0.4763175 , 0.44348219, 0.43760464, 0.46331373, 0.46059358,
       0.49193815, 0.45334611, 0.4148705 , 0.46012182, 0.49828153,
       0.45678312, 0.41289012, 0.3985481 , 0.54219818, 0.4237927 ,
       0.48253504, 0.48211182, 0.44458913, 0.40877816, 0.42871213,
       0.41068475, 0.43300114, 0.4475406 , 0.39970612, 0.46776432,
       0.48756183, 0.50319717, 0.49148173, 0.46290145, 0.41533673,
       0.47862241, 0.42337934, 0.42608224, 0.42957295, 0.45254181,
       0.51076857, 0.45129342, 0.41114481, 0.40030545, 0.47629941,
       0.43170991, 0.45192619, 0.49509341, 0.42241396, 0.39057422,
       0.45396656, 0.41669858, 0.41805785, 0.5100338 , 0.40930952,
       0.41500768, 0.41914857, 0.39072926, 0.43627077, 0.40783304,
       0.42898451, 0.40882082, 0.49418927, 0.42456477, 0.48661036,
       0.47998659, 0.42568324, 0.50402659, 0.40827109, 0.46671003,
       0.43891057, 0.45234879, 0.47069633, 0.49924173, 0.46359026,
       0.501231  , 0.40501515, 0.40418054, 0.42083781, 0.47488649,
       0.44139856, 0.42861878, 0.48184014, 0.42295036, 0.40842482,
       0.48090098, 0.44242891, 0.47936795, 0.48269666, 0.46264305,
       0.43254356, 0.39929742, 0.42062254, 0.44883376, 0.42137929,
       0.40549523, 0.46442782, 0.40963496, 0.42616499, 0.39842361,
       0.50647962, 0.42466782, 0.50804032, 0.41201602, 0.43970293,
       0.42845106, 0.46404743, 0.46410349, 0.41022695, 0.42101677]
auc_seed_42=[0.49279304, 0.47368064, 0.49683308, 0.51233485, 0.39607817,
       0.41688536, 0.45563531, 0.48045363, 0.41202241, 0.40671599,
       0.4638633 , 0.45028599, 0.43670305, 0.46852732, 0.39474342,
       0.41466582, 0.47977751, 0.45002138, 0.43067004, 0.4545595 ,
       0.45227587, 0.47971644, 0.44297335, 0.46412828, 0.41527541,
       0.42744823, 0.41098117, 0.49431357, 0.44537838, 0.47819887,
       0.43450002, 0.51936952, 0.45400242, 0.41108519, 0.43832643,
       0.40959441, 0.41334475, 0.40291161, 0.52172886, 0.41519642,
       0.39814339, 0.43060788, 0.50182257, 0.42340122, 0.53032753,
       0.4239218 , 0.41529837, 0.42448252, 0.40642301, 0.45077517,
       0.40493564, 0.39612967, 0.5076147 , 0.40077921, 0.48202241,
       0.4385266 , 0.50139991, 0.49945206, 0.44863544, 0.41259759,
       0.38623253, 0.5121619 , 0.50740744, 0.46261318, 0.44114574,
       0.44561241, 0.4396294 , 0.45274709, 0.47499801, 0.47895059,
       0.40847672, 0.5155511 , 0.45779239, 0.47636527, 0.44245285,
       0.42482229, 0.44112688, 0.43327315, 0.45119025, 0.45885774,
       0.42346301, 0.42367419, 0.45577199, 0.42195099, 0.41723304,
       0.41372757, 0.4692384 , 0.44297636, 0.40377551, 0.43819255,
       0.437552  , 0.51227415, 0.44017429, 0.47219729, 0.44447507,
       0.4499654 , 0.45790941, 0.42955708, 0.43367849, 0.45239827,
       0.44994281, 0.43551871, 0.42809633, 0.40903854, 0.41018485,
       0.4255747 , 0.43153496, 0.50659904, 0.49845768, 0.40914647,
       0.49632985, 0.42087384, 0.45256177, 0.58479341, 0.48103539,
       0.46091293, 0.48146724, 0.4990146 , 0.47283324, 0.45416367,
       0.43240318, 0.41424593, 0.45569847, 0.42913299, 0.41374497,
       0.46264059, 0.44131205, 0.45703206, 0.48177158, 0.44306042,
       0.49989456, 0.45807352, 0.50712242, 0.41282532, 0.43194015,
       0.44096559, 0.4332469 , 0.40734158, 0.43349578, 0.40225138,
       0.44149187, 0.40017272, 0.42838472, 0.45903045, 0.39864005,
       0.50165558, 0.41385676, 0.49818752, 0.4598325 , 0.49115999,
       0.43502867, 0.53923353, 0.41155483, 0.44681538, 0.44026353,
       0.39623896, 0.43288586, 0.465456  , 0.4452314 , 0.43254797,
       0.42512556, 0.43440016, 0.43058741, 0.4485551 , 0.45754094,
       0.3969528 , 0.44356926, 0.43510994, 0.43709334, 0.41953178,
       0.42615433, 0.54449456, 0.45179024, 0.51683531, 0.39942284,
       0.43199301, 0.40821941, 0.48104999, 0.38893118, 0.41806563,
       0.44559603, 0.42435804, 0.50682868, 0.43316919, 0.51103005,
       0.43976506, 0.45952639, 0.42954273, 0.43469739, 0.41651917,
       0.45821947, 0.44199006, 0.40671473, 0.42759907, 0.43686835,
       0.41826272, 0.44056638, 0.50138266, 0.46585807, 0.47231159]
auc_seed_52=[0.4047051 , 0.43699691, 0.45599761, 0.46590102, 0.43439436,
       0.40456231, 0.44480165, 0.44336503, 0.42568519, 0.41183809,
       0.46642371, 0.48084038, 0.48599051, 0.41263206, 0.41450804,
       0.4094887 , 0.41804891, 0.42610564, 0.43951516, 0.47543014,
       0.4608488 , 0.42297713, 0.52074905, 0.44414888, 0.41105149,
       0.48165973, 0.46679866, 0.44100907, 0.49933577, 0.46902588,
       0.42618711, 0.42211293, 0.40452257, 0.48033523, 0.41071886,
       0.42691582, 0.41455739, 0.39724218, 0.43463923, 0.46318951,
       0.39296699, 0.49974394, 0.41234958, 0.44797385, 0.3988093 ,
       0.42351757, 0.41354161, 0.40347979, 0.43313776, 0.4302388 ,
       0.50171585, 0.53488554, 0.49459495, 0.450394  , 0.38924131,
       0.46414883, 0.44051838, 0.42803353, 0.47995875, 0.40586865,
       0.43152372, 0.43301014, 0.42546591, 0.41636558, 0.43050129,
       0.41141681, 0.42776743, 0.4216716 , 0.4310597 , 0.42420368,
       0.42879743, 0.39861724, 0.42171919, 0.43459486, 0.43647261,
       0.413943  , 0.40047931, 0.409448  , 0.48042114, 0.43877287,
       0.47453263, 0.40321851, 0.45585098, 0.45832823, 0.43239587,
       0.42996521, 0.47149867, 0.47159891, 0.41164301, 0.42841541,
       0.44583511, 0.50451287, 0.40672727, 0.41486663, 0.45813757,
       0.44660733, 0.40862667, 0.50753656, 0.47682663, 0.49768785,
       0.41145528, 0.45658212, 0.44813082, 0.39632881, 0.4083087 ,
       0.45651602, 0.4438443 , 0.41372632, 0.44075332, 0.45727886,
       0.44537042, 0.45374599, 0.46349677, 0.40466857, 0.46430614,
       0.4846924 , 0.46219676, 0.51852413, 0.4407187 , 0.47792152,
       0.42614054, 0.44895946, 0.43699076, 0.41890768, 0.4659656 ,
       0.41401487, 0.40371223, 0.4012073 , 0.49367476, 0.50057118,
       0.40175613, 0.41617701, 0.42519144, 0.45342528, 0.41326056,
       0.43272363, 0.43013319, 0.43121358, 0.40924674, 0.43584061,
       0.41708848, 0.45297271, 0.40715702, 0.44172625, 0.41350653,
       0.40171508, 0.4463458 , 0.54324707, 0.41725826, 0.4009898 ,
       0.42998039, 0.43892588, 0.41456766, 0.40790414, 0.50598226,
       0.4743991 , 0.41141253, 0.42713354, 0.40277649, 0.46355866,
       0.4770698 , 0.4114004 , 0.41685923, 0.44408609, 0.50441144,
       0.44218876, 0.42194487, 0.43025692, 0.45474703, 0.39764681,
       0.44633938, 0.39136967, 0.46778528, 0.40133008, 0.50660332,
       0.41041816, 0.49590925, 0.39983843, 0.41145782, 0.40424993,
       0.42961056, 0.44660215, 0.41298173, 0.45418405, 0.43750715,
       0.46774176, 0.45867654, 0.47731224, 0.41758992, 0.46367886,
       0.4310798 , 0.45869796, 0.45166032, 0.39971615, 0.40794748,
       0.49352092, 0.48279345, 0.43015266, 0.42048091, 0.40665733]
auc_seed_62=[0.42930227, 0.50676979, 0.46457575, 0.44290574, 0.43917501,
       0.4284849 , 0.39338025, 0.43548626, 0.41659428, 0.41580596,
       0.4334514 , 0.42501462, 0.38280643, 0.52821541, 0.42714482,
       0.49909452, 0.44161652, 0.40691376, 0.42762913, 0.45616394,
       0.44251924, 0.43986697, 0.48363996, 0.45176765, 0.42767502,
       0.39755381, 0.42456922, 0.42623748, 0.45784684, 0.47096148,
       0.49972001, 0.47081588, 0.47000507, 0.43405545, 0.39239005,
       0.4618662 , 0.47793189, 0.42243393, 0.50641273, 0.41425829,
       0.42653691, 0.45881083, 0.45098804, 0.59187774, 0.48369474,
       0.4147465 , 0.40474974, 0.46753654, 0.42490366, 0.41401024,
       0.44985844, 0.49031076, 0.45229085, 0.42334238, 0.40436916,
       0.43521908, 0.42746268, 0.42545661, 0.46457439, 0.49736119,
       0.46783392, 0.45606007, 0.45698141, 0.40690265, 0.45924091,
       0.49781079, 0.39811789, 0.41768728, 0.42621621, 0.49863035,
       0.45862328, 0.45517089, 0.43024312, 0.46134536, 0.50757354,
       0.43897887, 0.49908188, 0.43768879, 0.43293701, 0.47533041,
       0.40934285, 0.44345318, 0.45406899, 0.49842325, 0.42160474,
       0.49475495, 0.46209796, 0.46420758, 0.4595799 , 0.40479082,
       0.50355666, 0.45044212, 0.39447563, 0.44472944, 0.45975575,
       0.49834803, 0.4062195 , 0.44191481, 0.46243639, 0.50102002,
       0.42940286, 0.43552556, 0.4591373 , 0.44168525, 0.43536232,
       0.46356349, 0.48068655, 0.50345322, 0.42665691, 0.42546426,
       0.41026555, 0.44661694, 0.42651541, 0.4373423 , 0.48037944,
       0.4273906 , 0.408189  , 0.44347461, 0.44869147, 0.4240215 ,
       0.47872966, 0.45566626, 0.47418842, 0.49568408, 0.42914832,
       0.45165236, 0.50537524, 0.42257878, 0.45695922, 0.416692  ,
       0.3992001 , 0.40821481, 0.43807436, 0.49695843, 0.42748657,
       0.42576833, 0.40235898, 0.43942031, 0.39724353, 0.46661244,
       0.50385096, 0.4943782 , 0.51129109, 0.42959931, 0.40914724,
       0.45958264, 0.42546224, 0.43870219, 0.43953953, 0.40082384,
       0.42198754, 0.4429749 , 0.42294232, 0.4323872 , 0.40176492,
       0.45625189, 0.4494987 , 0.44354709, 0.43105188, 0.43768777,
       0.45757652, 0.45855209, 0.48783008, 0.40868506, 0.41090058,
       0.48607925, 0.40485601, 0.41743565, 0.45446491, 0.44179515,
       0.41711711, 0.44097161, 0.39481893, 0.41593829, 0.42706805,
       0.44697945, 0.41927398, 0.41820631, 0.39657084, 0.42548235,
       0.44221481, 0.48208   , 0.4414051 , 0.45430497, 0.47914938,
       0.42995263, 0.41683557, 0.48614374, 0.42834056, 0.4527078 ,
       0.45136008, 0.43751561, 0.39917065, 0.51843503, 0.40408806,
       0.49635138, 0.49317808, 0.42261358, 0.44417262, 0.43572057]
auc_seed_72=[0.43859101, 0.43599145, 0.46381968, 0.45135742, 0.46190539,
       0.43747692, 0.41330919, 0.40450779, 0.44834312, 0.39248949,
       0.40097047, 0.43721528, 0.47654267, 0.41289407, 0.42019885,
       0.43961936, 0.41477974, 0.45750765, 0.41191168, 0.43821442,
       0.44129561, 0.43211626, 0.41285634, 0.44107847, 0.42353719,
       0.4250157 , 0.45825672, 0.43890167, 0.42672057, 0.49796513,
       0.45237899, 0.45354898, 0.43961052, 0.50099923, 0.52349105,
       0.43242104, 0.54174358, 0.47105122, 0.43680472, 0.4661395 ,
       0.45537572, 0.51046171, 0.42333842, 0.49519548, 0.42578382,
       0.51843305, 0.45458623, 0.41023929, 0.47759856, 0.45398621,
       0.43168304, 0.45775748, 0.4431632 , 0.4707206 , 0.41598054,
       0.43135734, 0.44754151, 0.4049953 , 0.44027724, 0.40376842,
       0.45985128, 0.44969287, 0.43884169, 0.48617261, 0.50683989,
       0.48363148, 0.4054689 , 0.48749371, 0.43899894, 0.4032715 ,
       0.41814857, 0.42515829, 0.40338351, 0.46020343, 0.4568116 ,
       0.42451458, 0.4328973 , 0.43375973, 0.51599306, 0.466439  ,
       0.46260746, 0.42479103, 0.49021134, 0.47251123, 0.4537859 ,
       0.45716564, 0.45370413, 0.4941034 , 0.43556372, 0.43025912,
       0.41259363, 0.39780001, 0.50799938, 0.49187122, 0.44887406,
       0.43962175, 0.47730629, 0.47190274, 0.48297286, 0.46374617,
       0.44170341, 0.43480218, 0.41145609, 0.4091292 , 0.47502928,
       0.41031764, 0.50267382, 0.42384486, 0.42488131, 0.43820144,
       0.40629481, 0.42377139, 0.43454662, 0.46176913, 0.40936225,
       0.45684945, 0.4228256 , 0.51361614, 0.4532827 , 0.43615216,
       0.39591837, 0.47936548, 0.40615948, 0.41557911, 0.42621663,
       0.42114315, 0.45929825, 0.41923586, 0.42755706, 0.44962398,
       0.48886792, 0.4299686 , 0.50719442, 0.4019318 , 0.45686359,
       0.41132436, 0.50644306, 0.45709986, 0.50491717, 0.4056907 ,
       0.46005551, 0.40725337, 0.51760452, 0.50988153, 0.44998452,
       0.5195468 , 0.44048471, 0.44646572, 0.41429453, 0.47973613,
       0.45563986, 0.4008133 , 0.49360749, 0.48979205, 0.49994455,
       0.47597601, 0.43803649, 0.46155809, 0.44269851, 0.43581733,
       0.44050572, 0.50753991, 0.4242907 , 0.47935283, 0.49532649,
       0.4125062 , 0.4121948 , 0.41072667, 0.50892631, 0.44738843,
       0.42929899, 0.52213403, 0.48155787, 0.4835701 , 0.44256062,
       0.46365364, 0.48482731, 0.41814962, 0.42388667, 0.48956916,
       0.43719409, 0.39348822, 0.45077581, 0.44473036, 0.4279082 ,
       0.42781156, 0.40644755, 0.45930452, 0.46285772, 0.43472887,
       0.43661927, 0.45518484, 0.39506157, 0.485805  , 0.40482144,
       0.5086741 , 0.420169  , 0.45671473, 0.44168477, 0.41257744]
auc_seed_82=[0.43710806, 0.49451169, 0.4346535 , 0.48463734, 0.502831  ,
       0.42267946, 0.41771937, 0.47404628, 0.42291573, 0.4255155 ,
       0.42774017, 0.49235541, 0.46635963, 0.46477046, 0.46127859,
       0.42421444, 0.42623256, 0.4146678 , 0.42815513, 0.46017157,
       0.43939348, 0.44308545, 0.46753943, 0.44185532, 0.47602734,
       0.47017138, 0.47892082, 0.41076184, 0.44685363, 0.50053479,
       0.44163528, 0.48028617, 0.40784298, 0.41010294, 0.51179894,
       0.4267973 , 0.43569164, 0.45230727, 0.44801509, 0.43038667,
       0.38774968, 0.46286232, 0.42685231, 0.42184784, 0.42320478,
       0.40864803, 0.48535832, 0.43206528, 0.40689239, 0.40039829,
       0.46703567, 0.4383495 , 0.58535649, 0.42588716, 0.39791721,
       0.48292661, 0.43525191, 0.40781268, 0.48910058, 0.45644715,
       0.54914666, 0.41541872, 0.43603432, 0.41810961, 0.45187573,
       0.52007915, 0.4337329 , 0.43537882, 0.39276961, 0.43900018,
       0.48425836, 0.44388489, 0.45968485, 0.4854533 , 0.45732196,
       0.45447961, 0.41590829, 0.41537012, 0.42097848, 0.54049827,
       0.42022543, 0.44441613, 0.42746324, 0.47160529, 0.49696527,
       0.45702762, 0.42475531, 0.42284232, 0.4989622 , 0.4913078 ,
       0.46355756, 0.50700128, 0.41549321, 0.46059426, 0.43793367,
       0.49784484, 0.47009819, 0.44532875, 0.52844969, 0.54644998,
       0.42810396, 0.41622702, 0.43002069, 0.40847856, 0.47890689,
       0.4567264 , 0.49394102, 0.40825358, 0.50938773, 0.43108609,
       0.46652645, 0.39951469, 0.47429682, 0.50456272, 0.4709481 ,
       0.43504131, 0.4568774 , 0.40699536, 0.43165988, 0.4305072 ,
       0.43217267, 0.43571302, 0.50350454, 0.414952  , 0.49596142,
       0.4454651 , 0.47699718, 0.45922575, 0.50204783, 0.43228545,
       0.41294155, 0.44175555, 0.47656834, 0.42393229, 0.50138484,
       0.50668002, 0.40883829, 0.45554673, 0.43735111, 0.43189536,
       0.49955611, 0.4125218 , 0.40800492, 0.39193252, 0.48160968,
       0.42923649, 0.40456402, 0.49374957, 0.52144307, 0.50079201,
       0.46961277, 0.40746594, 0.4781193 , 0.44918117, 0.49866607,
       0.49005182, 0.50737626, 0.49288232, 0.46227121, 0.41977476,
       0.40854799, 0.41246444, 0.49852024, 0.43502025, 0.44890894,
       0.4588709 , 0.42730923, 0.47991691, 0.44982512, 0.43407805,
       0.47305542, 0.45628369, 0.39705672, 0.40120042, 0.45533289,
       0.42073751, 0.44133924, 0.40686466, 0.46191353, 0.44573801,
       0.44569496, 0.39317936, 0.41241398, 0.50710096, 0.45390361,
       0.46882207, 0.44718067, 0.42907509, 0.4356466 , 0.52004349,
       0.4064971 , 0.45482235, 0.42823218, 0.42072816, 0.48029178,
       0.41644265, 0.43073236, 0.42613835, 0.47411467, 0.42352577]
auc_seed_92=[0.50810015, 0.44375742, 0.40809207, 0.43143116, 0.50062313,
       0.5015563 , 0.39835973, 0.43609766, 0.40635871, 0.45614772,
       0.47346789, 0.42238157, 0.44633946, 0.40351261, 0.43930531,
       0.50886295, 0.43333277, 0.388699  , 0.41461171, 0.42674024,
       0.46049233, 0.43955722, 0.4458036 , 0.42497189, 0.48130721,
       0.45175425, 0.49775665, 0.46069769, 0.47031084, 0.41714943,
       0.3971374 , 0.43779643, 0.41755165, 0.45564793, 0.44620267,
       0.5001704 , 0.4349777 , 0.47276485, 0.44846997, 0.41318705,
       0.42531086, 0.50605159, 0.42765838, 0.42811973, 0.43407095,
       0.41404919, 0.41662068, 0.42768989, 0.49138166, 0.48076804,
       0.41128792, 0.42298143, 0.41729152, 0.46331331, 0.40573704,
       0.40590794, 0.41089136, 0.5018994 , 0.43806636, 0.41581833,
       0.42787525, 0.41920403, 0.45493627, 0.51091666, 0.45599167,
       0.43546171, 0.45620965, 0.49500204, 0.42043283, 0.40797168,
       0.44969035, 0.39234338, 0.40986522, 0.42661945, 0.43594017,
       0.39244261, 0.43608708, 0.44421839, 0.54547987, 0.43584281,
       0.43870492, 0.44033058, 0.43835625, 0.46975827, 0.44056819,
       0.42888153, 0.47898041, 0.49551006, 0.44454518, 0.50255249,
       0.44473816, 0.44932396, 0.42572791, 0.4355733 , 0.43833718,
       0.41858328, 0.51019654, 0.43773288, 0.43468978, 0.4554676 ,
       0.38826754, 0.46425778, 0.4275144 , 0.45185169, 0.42224674,
       0.44056792, 0.43683156, 0.44598982, 0.4337574 , 0.39359091,
       0.44128636, 0.43625859, 0.50170586, 0.46418357, 0.42700947,
       0.47701343, 0.4440411 , 0.45556435, 0.42492568, 0.45314483,
       0.40787608, 0.4317345 , 0.4530498 , 0.43689517, 0.41464298,
       0.45382012, 0.40299326, 0.38958569, 0.43209882, 0.43161293,
       0.41795028, 0.43157595, 0.42617543, 0.45731427, 0.44535566,
       0.45019978, 0.45471101, 0.4534236 , 0.49583468, 0.44647422,
       0.39694648, 0.49802446, 0.45744447, 0.47507242, 0.41751839,
       0.45552461, 0.45744678, 0.51781939, 0.42911151, 0.43321968,
       0.42579045, 0.48595655, 0.41728741, 0.43768831, 0.48512275,
       0.42259493, 0.39654556, 0.39439929, 0.40516789, 0.40943751,
       0.40831143, 0.48236081, 0.42525502, 0.43266918, 0.41684466,
       0.44219482, 0.52614234, 0.45708559, 0.40222959, 0.42643153,
       0.40549426, 0.39171313, 0.53974806, 0.42039248, 0.45339833,
       0.44307713, 0.45630678, 0.4813349 , 0.43181986, 0.45742107,
       0.4753139 , 0.44449852, 0.40830404, 0.41940082, 0.4677356 ,
       0.4333877 , 0.39082595, 0.46725904, 0.48856152, 0.4105513 ,
       0.57512129, 0.49641051, 0.42775836, 0.41950002, 0.41877583,
       0.45131918, 0.39843107, 0.41442897, 0.41677696, 0.46413677]

auc_best_score_mean=0.575
auc_best_score_std=0.025
exec_time_mean=1806