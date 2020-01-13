import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def corr(data1,data2):

    diff = data1-data2
    print(diff)
    m=np.mean(diff)
    data3=data1-np.array(tile(m,(len(data1))))
    print(data3)
    m2=np.mean(data3-data2)

    print(sum(data3-data2))
    # print(m)
    # m=np.array(m)
    diff2=np.array(tile(m,(len(data1))))-diff
    # print(tile(m,(len(data1))))
    sdiff=diff2**2
    sd=sdiff.sum()
    s=sd/(len(data2)-1)
    s=s**0.5
    m+=m2

    return s,m

if __name__=='__main__':
    # data1=[18.0774, 18.0872, 18.0774, 18.0676, 18.0774, 18.0872, 18.0872, 18.0872, 18.1068, 18.1068]
    # data2=[18.8418, 18.832, 18.832, 18.8516, 18.8418, 18.832, 18.832, 18.832, 18.8418, 18.8418]
    data1=[31.1408, 31.1996, 31.278, 31.2976, 31.3172, 31.3956, 31.3564, 31.3074, 31.2682, 31.18, 31.2094, 31.425, 31.4642, 31.523, 31.6308, 31.7092, 31.6602, 31.7484, 31.7386, 31.7288, 31.768, 31.8072, 31.7778, 31.9248, 31.817, 31.7288, 31.7582, 31.5818, 31.4642, 31.474, 31.3172, 31.2682, 31.278, 31.2388, 31.278, 31.0428, 31.0428, 31.0722, 31.2388, 31.2192, 31.2682, 31.425, 31.4348, 31.5328, 31.4054, 31.474, 31.5034, 31.6504, 31.5818, 31.5034]
    data2=[29.4258, 29.416, 29.4356, 29.3866, 29.563, 29.5532, 29.612, 29.6316, 29.6316, 29.7296, 29.7688, 29.7982, 29.8276, 29.8864, 29.8472, 29.857, 29.8472, 29.7982, 29.8276, 29.7492, 29.7198, 29.7198, 29.6218, 29.6512, 29.6022, 29.4944, 29.4748, 29.4356, 29.416, 29.4062, 29.2298, 29.1906, 29.2102, 29.22, 29.2102, 29.2984, 29.3082, 29.3474, 29.4356, 29.4552, 29.4748, 29.5238, 29.563, 29.6022, 29.6512, 29.5042, 29.4846, 29.5532, 29.5532, 29.6218]
    print(pearsonr(data1, data2))
    data1=np.array(data1)
    data2 = np.array(data2)

    # print(data1-data2)
    corr, m = corr(data1, data2)

    zaosheng2=(np.random.normal(0,corr,size=len(data1)))
    zaosheng1=0
    pre_data2=data1-m+zaosheng1
    pre_data22 = data1 - m + zaosheng2
    # data2=np.around(data2, decimals=1)
    # pre_data2=np.around(pre_data2, decimals=1)
    # print(data1,data2,pre_data2)
    print(corr,m)
    # plt.scatter(data1,data2)

    c = {"data1": data1,"data2": data2, "pre_data2": pre_data2,"pre_data22": pre_data22}
    data = pd.DataFrame(c)
    data.plot()
    plt.show()
