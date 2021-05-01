import matplotlib.pyplot as plt
seeds=[1,12,22,32,42,52,62,72,82,92]

random_best_auc=[0.53372,0.59024,0.60851,0.59343,0.58479,0.54324,0.59187,0.54174,0.58535,0.57512]

bayes_best_auc=[0.60922,0.61161,0.60944,0.60984,0.61323,0.61444,0.60483,0.61145,0.60903,0.61152]
plt.figure(figsize=(20,5))
plt.xticks(seeds)
plt.plot(seeds,random_best_auc, color='blue')       
plt.plot(seeds,bayes_best_auc, color='g')   

for i, txt in enumerate(random_best_auc):
    plt.annotate(round(txt,3), (seeds[i], random_best_auc[i]-0.005),size=15)
plt.annotate(round(random_best_auc[0],3), (seeds[0], random_best_auc[0]),size=15)

for i, txt in enumerate(bayes_best_auc):
    plt.annotate(round(txt,3), (seeds[i], bayes_best_auc[i]),size=15)

plt.xlabel("Seed",fontsize=20)
plt.ylabel("AUC",fontsize=20)