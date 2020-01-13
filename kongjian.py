import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def getdata():

    filename = 'C:/Users/windows/Downloads/data.txt/data.txt'
    date1 = []
    time1 = []
    epoch1 = []
    moteid1 = []
    temperature1 = []
    humidity1 = []
    light1 = []
    voltage1 = []

    date2 = []
    time2 = []
    epoch2 = []
    moteid2 = []
    temperature2 = []
    humidity2 = []
    light2 = []
    voltage2 = []
    count = 1
    with open(filename, 'r') as file_to_read:
        while(True):
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            if len(lines.split()) != 8:
                pass
            else:
                d_tmp, t_tmp, e_tmp, m_tmp, te_tmp, h_tmp, l_tmp, v_tmp = [i for i in lines.split()]
                if d_tmp == '2004-03-01'and m_tmp=='22':
                    date1.append(d_tmp)
                    time1.append(t_tmp)
                    epoch1.append(e_tmp)
                    moteid1.append(m_tmp)
                    temperature1.append(te_tmp)
                    humidity1.append(h_tmp)
                    light1.append(l_tmp)
                    voltage1.append(v_tmp)
                if d_tmp == '2004-03-01'and m_tmp=='23':
                    date2.append(d_tmp)
                    time2.append(t_tmp)
                    epoch2.append(e_tmp)
                    moteid2.append(m_tmp)
                    temperature2.append(te_tmp)
                    humidity2.append(h_tmp)
                    light2.append(l_tmp)
                    voltage2.append(v_tmp)
                else:
                    pass
                count = count + 1
            pass


    data1 = list(map(float, temperature1))
    data2=list(map(float,temperature2))
    return data1,data2

if __name__=='__main__':
    data1,data2=getdata()
    print(len(data1),data1)
    print(len(data2),data2)
    # print(data2[274])
    m=min(len(data1),len(data2))
    c = {"data1": data1[1500:1550], "data2": data2[1500:1550]}
    data = pd.DataFrame(c)
    # plt.subplot(211)
    data.plot()
    # plt.subplot(212)
    # plt.xlim(30, 35)
    # plt.ylim(25, 30)
    print(pearsonr(data1[1500:1550], data2[1500:1550]))
    print(data1[1500:1550])
    print(data2[1500:1550])
    # plt.scatter(data1[:20], data2[:20])

    plt.show()