import matplotlib.pyplot as plt

def bar_chart(x,y,x_label,y_label,title,decimal,locations):
    # barchart with flexible text locations and color
    bars_1=plt.bar(x,y)
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,y1,1.1*y2))
    for bar,l in zip(bars_1,locations):
        height=bar.get_height()
        warna="black"
        plt.text(bar.get_x() + bar.get_width()/2., l*height,
                    decimal % float(height),
                    ha='center', va='bottom',color=warna)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.title(title,fontsize=20)
    plt.tight_layout()
    plt.show()

def scatter_subplot(x,y_list,x_label,y_label,best_indexes,titles):
    # untuk grid search
    # satu bintang
    star_color='yellow'
    start_size=22
    y_count=len(y_label)
    fig,axs=plt.subplots(y_count,1,figsize=(20,15)) 

    for i,y in enumerate(y_list):
        axs[i].scatter(x,y)
        x1,x2,y1,y2 = axs[i].axis()  
        axs[i].axis((x1,x2,y1,1.01*y2))
        best_index=best_indexes[i]
        axs[i].plot(x[best_index],y[best_index],marker='*',markersize=start_size,markerfacecolor=star_color)
        x_text=str(best_index)
        y_text=str(round(y[best_index],3))
        axs[i].text(best_index+4,y[best_index]-0.001,"("+x_text+","+y_text+")",fontsize=20)
        axs[i].set_title(titles[i],size=20,loc="left")
        axs[i].set_ylabel(y_label,fontsize=20)

    axs[i].set_xlabel(x_label,fontsize=20)

def scatter_subplot_2(x,y_list,x_label,y_label,best_indexes,titles):
    # bayes vs random
    # dua bintang
    star_bayes_color='green'
    star_random_color='yellow'
    star_size=22
    y_count=len(y_label)
    fig,axs=plt.subplots(y_count,1,figsize=(20,15)) 

    for i,y in enumerate(y_list):
        axs[i].scatter(x,y[0])
        axs[i].scatter(x,y[1])
        best_index_r=best_indexes[i][0]
        best_index_b=best_indexes[i][1]
        axs[i].plot(x[best_index_r],y[0][best_index_r],marker='*',markersize=star_size,markerfacecolor=star_random_color)
        axs[i].plot(x[best_index_b],y[1][best_index_b],marker='*',markersize=star_size,markerfacecolor=star_bayes_color)
        axs[i].set_title(titles[i],size=20,loc="left")
        axs[i].set_ylabel(y_label,fontsize=20)

    axs[i].set_xlabel(x_label,fontsize=20)


def scatter_subplot_3():
    # menunjukkan semua hasil bayes vs random untuk tiap seed. 
    # jaga2 aja kalau misalkan nanti diperlukan 
    pass