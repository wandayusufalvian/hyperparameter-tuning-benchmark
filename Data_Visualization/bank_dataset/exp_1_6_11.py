import matplotlib.pyplot as plt

gbdt = ["xgboost","lightgbm","catboost"]
auc = [0.412,0.409,0.603]
std_auc =[0.163,0.172,0.08]

time=[2.694,0.708,7.014]
std_time=[0.026,0.011,0.12]

bars_1=plt.bar(gbdt,auc)

for bar in bars_1:
    height=bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 0.9*height,
                '%.3f' % float(height),
                ha='center', va='bottom',color='white')
fig=plt.xlabel('GBDT',fontsize=20)
fig=plt.ylabel('AUC',fontsize=20)
plt.title("Performance",fontsize=20)
plt.show()

bars_2=plt.bar(gbdt,time)


bar=bars_2[0]
height=bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                '%.3f' % float(height),
                ha='center', va='bottom',color='black')

bar=bars_2[1]
height=bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                '%.3f' % float(height),
                ha='center', va='bottom',color='black')


bar=bars_2[2]
height=bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., 0.9*height,
                '%.3f' % float(height),
                ha='center', va='bottom',color='white')

fig=plt.xlabel('GBDT',fontsize=20)
fig=plt.ylabel('Time(Seconds)',fontsize=20)
plt.title("Time",fontsize=20)
plt.show()