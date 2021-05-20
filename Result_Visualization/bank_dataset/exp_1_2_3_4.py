y=[0.412,0.523,0.575,0.61]


x=["no-HPO","grid search","random search","bayes search"]

import matplotlib.pyplot as plt

bars_1=plt.bar(x,y)


for bar in bars_1:
    height=bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 0.9*height,
                '%.3f' % float(height),
                ha='center', va='bottom',color='white')
fig=plt.xlabel('HPO',fontsize=20)
fig=plt.ylabel('AUC',fontsize=20)
plt.title("Performance",fontsize=20)
plt.show()

y=[1693,1806,1281]


x=["grid search","random search","bayes search"]

import matplotlib.pyplot as plt

bars_1=plt.bar(x,y)


for bar in bars_1:
    height=bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 0.9*height,
                '%.0f' % float(height),
                ha='center', va='bottom',color='white')
fig=plt.xlabel('HPO',fontsize=20)
fig=plt.ylabel('Time(seconds)',fontsize=20)
plt.title("Time",fontsize=20)
plt.show()



