#===========bar chart waktu============
import matplotlib.pyplot as plt



labels=["xgboost","lightgbm","catboost"]
time=[1693,55,4839]




bars_2=plt.bar(labels,time)

bar=bars_2[0]
height=bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                '%.0f' % float(height),
                ha='center', va='bottom',color='black')

bar=bars_2[1]
height=bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                '%.0f' % float(height),
                ha='center', va='bottom',color='black')


bar=bars_2[2]
height=bar.get_height()
plt.text(bar.get_x() + bar.get_width()/2., 0.9*height,
                '%.0f' % float(height),
                ha='center', va='bottom',color='white')

fig=plt.xlabel('GBDT',fontsize=20)
fig=plt.ylabel('Time(Seconds)',fontsize=20)
plt.show()