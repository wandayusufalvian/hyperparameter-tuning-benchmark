import matplotlib.pyplot as plt

def bar_chart(x,y,x_label,y_label,title,decimal):
    bars_1=plt.bar(x,y)


    for bar in bars_1:
        height=bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 0.9*height,
                    decimal % float(height),
                    ha='center', va='bottom',color='white')
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.title(title,fontsize=20)
    plt.show()


def bar_chart_2(x,y,x_label,y_label,title,decimal,locations):
    bars_1=plt.bar(x,y)


    for bar,l in zip(bars_1,locations):
        height=bar.get_height()
        warna="white"
        if(l>1):
            warna="black"
        plt.text(bar.get_x() + bar.get_width()/2., l*height,
                    decimal % float(height),
                    ha='center', va='bottom',color=warna)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.title(title,fontsize=20)
    plt.show()