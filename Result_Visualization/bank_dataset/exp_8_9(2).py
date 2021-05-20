#========================AUC score=====================================================


x_random_best_auc=[0.53372,0.59024,0.60851,0.59343,0.58479,0.54324,0.59187,0.54174,0.58535,0.57512]
x_bayes_best_auc=[0.60922,0.61161,0.60944,0.60984,0.61323,0.61444,0.60483,0.61145,0.60903,0.61152]

l_random_best_auc=[0.52719,0.53319,0.50106,0.53611,0.52831,0.52446,0.52637,0.53035,0.52534,0.53298]
l_bayes_best_auc=[0.52679,0.52661,0.53917,0.52863,0.52883,0.52628,0.52766,0.52896,0.52974,0.52926]

c_random_best_auc=[0.64699,0.6468,0.65501,0.64959,0.65088,0.6498,0.64933,
0.64966,0.65023,0.65248]
c_bayes_best_auc=[0.65337,0.65309,0.65202,0.65158,0.65439,0.6525,0.65399,
0.65752,0.65998,0.65373]



seeds=[1,12,22,32,42,52,62,72,82,92]
# ========================Visualisasi==================================================

import matplotlib.pyplot as plt

x=range(1,200+1)
star_color='yellow'
start_size=22

fig,axs=plt.subplots(3,1,figsize=(20,15))

axs[0].plot(seeds,x_random_best_auc, color='blue') 
axs[0].plot(seeds,x_bayes_best_auc, color='g') 
axs[0].set_xticks(seeds)

axs[0].text(88,0.534,"XGBoost",fontsize=20)

axs[0].set_ylabel('AUC',fontsize=20)

for i, txt in enumerate(x_random_best_auc):
    axs[0].annotate(round(txt,3), (seeds[i], x_random_best_auc[i]-0.005),size=15)
axs[0].annotate(round(x_random_best_auc[0],3), (seeds[0], x_random_best_auc[0]),size=15)

for i, txt in enumerate(x_bayes_best_auc):
    axs[0].annotate(round(txt,3), (seeds[i], x_bayes_best_auc[i]),size=15)


axs[1].plot(seeds,l_random_best_auc, color='blue') 
axs[1].plot(seeds,l_bayes_best_auc, color='g') 
axs[1].set_xticks(seeds)

axs[1].text(88,0.500,"LightGBM",fontsize=20)

axs[1].set_ylabel('AUC',fontsize=20)

for i, txt in enumerate(l_random_best_auc):
    axs[1].annotate(round(txt,3), (seeds[i], l_random_best_auc[i]-0.005),size=15)
axs[1].annotate(round(l_random_best_auc[0],3), (seeds[0], l_random_best_auc[0]),size=15)

for i, txt in enumerate(l_bayes_best_auc):
    axs[1].annotate(round(txt,3), (seeds[i], l_bayes_best_auc[i]),size=15)




















