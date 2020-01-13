# # -*- coding: utf-8 -*-
from pandas import Series
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import scipy as sp
from scipy.optimize import leastsq
import json
import random
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
        while count <= 3053:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            if len(lines.split()) != 8:
                pass
            else:
                d_tmp, t_tmp, e_tmp, m_tmp, te_tmp, h_tmp, l_tmp, v_tmp = [i for i in lines.split()]
                if d_tmp == '2004-02-28':
                    date1.append(d_tmp)
                    time1.append(t_tmp)
                    epoch1.append(e_tmp)
                    moteid1.append(m_tmp)
                    temperature1.append(te_tmp)
                    humidity1.append(h_tmp)
                    light1.append(l_tmp)
                    voltage1.append(v_tmp)
                if d_tmp == '2004-02-29':
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

    data = list(map(float, temperature1))  # temperature1是字符串，将其转换成浮点型数值
    return data
# 数据准备好了
# 初值为S2_1
def esmoothpredict_withskip(data):
    # Lenght=100#预测100个时间片的值

    a = 0.4
    #
    pre = []  # 预测值
    s1 = []
    s2 = []
    At = []
    Bt = []
    x = 0

    for n in range(0, 3):
        x = x + data[n]
    x = x / 3
    # 初始化0，1两个点的值
    pre.append(data[0])
    pre.append(data[1])
    s1.append(x)
    s2.append(x)
    s1.append(a * data[1] + (1 - a) * s1[0])
    s2.append(a * s1[1] + (1 - a) * s2[0])
    At.append(data[0])
    Bt.append(1)
    At.append(2 * s1[0] - s2[0])
    Bt.append((a / (1 - a)) * (s1[0] - s2[0]))

    thead=0.15

    sen_count = 0#感知数据次数
    com_count=0#传输数据次数
    MSE = 0#方差
    continu = []#连续跳过几个点
    k = 0  # 表示间隔，k=0表示要强制采样，>0表示不需采样
    Lk = 0
    for i in range(1, len(data)-1):  # 站1 望2
        At.append(2 * s1[i] - s2[i])
        Bt.append((a / (1 - a)) * (s1[i] - s2[i]))
        pre_tmp = At[i + 1] + Bt[i + 1]

        if k > 0:
            pre.append(pre_tmp)
            # s1.append(a * pre[i + 1] + (1 - a) * s1[i])
            # s2.append(a * s1[i + 1] + (1 - a) * s2[i])
            k -= 1
        else:
            sen_count+=1#感知数据但不一定传输
            continu.append(Lk)  # 记录连续几次没有传输数据
            # pre.append(data[i + 1])
            if abs(pre_tmp - data[i + 1]) < thead:#当满足条件时，接着用这次预测值做真实值预测下一次
                Lk += 1
                k = Lk
                pre.append(pre_tmp)
            else:
                com_count += 1#不满足条件，要传输数据，使用真实值预测下一时刻数据
                Lk = 0
                k = Lk
                pre.append(data[i+1])
        s1.append(a * pre[i + 1] + (1 - a) * s1[i])
        s2.append(a * s1[i + 1] + (1 - a) * s2[i])

    #     MSE = (pre[i] - data[i]) ** 2 + MSE
    # MSE=(pre[-1]-data[-1])**2+MSE
    # MSE = (MSE ** (1 / 2)) / int(len(data) - 2)
    return pre,sen_count,com_count
    #参数暂不需要，MSE不用在这个函数中求
    # return a,pre,MSE,continu,sen_count,com_count
def esmoothpredict_withoutskip(data):

    a = 0.4#a=0.4效果最好
    #
    pre = []  # 预测值调整过的
    pre2=[]#不调整预测值
    s1 = []
    s2 = []
    At = []
    Bt = []
    x = 0
    thead = 0.15

    for n in range(0, 3):
        x = x + data[n]
    x = x / 3
    # 初始化0，1两个点的值
    pre.append(data[0])
    pre.append(data[1])
    pre2.append(data[0])
    pre2.append(data[1])
    s1.append(x)
    s2.append(x)
    s1.append(a * data[1] + (1 - a) * s1[0])
    s2.append(a * s1[1] + (1 - a) * s2[0])
    At.append(data[0])
    Bt.append(1)
    At.append(2 * s1[0] - s2[0])
    Bt.append((a / (1 - a)) * (s1[0] - s2[0]))

    # sen_count = 0  # 感知数据次数
    com_count = 0  # 传输数据次数
    MSE = 0  # 方差
    for i in range(1,len(data)-1):#站1 望2
        # print(i)
        # print(At)
        At.append(2*s1[i]-s2[i])
        Bt.append((a / (1 - a))*(s1[i]-s2[i]))
        pre_tmp2=At[i+1]+Bt[i+1]
        pre2.append(pre_tmp2)
        if abs(pre_tmp2 - data[i + 1]) <thead:
            pre.append(pre_tmp2)
        else:
            com_count+=1
            pre.append(data[i + 1])
        s1.append(a * pre[i + 1] + (1 - a) * s1[i])
        s2.append(a * s1[i + 1] + (1 - a) * s2[i])
    #     MSE = (pre[i]- data[i]) ** 2 + MSE
    # MSE = (MSE ** (1 / 2)) / int(len(data)-1)
    # return a,pre,pre2,MSE,com_count
    return pre, com_count
#
def Mse(data1,data2):
    return sum([(d1-d2)**2 for d1,d2 in zip(data1,data2)])/len(data1)
def DBP_Model(data):
    W = len(data)  # 窗口大小10
    l = 3  # 边缘3
    ds = 0
    de = 0
    for i in range(l):
        ds += data[i]
        de += data[W - 1 - i]
    k = (de - ds) / ((W - l) * l)
    return k

def DBP_withskip(data):
    g = 0  # 表示间隔，k=0表示要强制采样，>0表示不需采样
    Lk = 0
    pre = data[0:10]
    thead = 0.15
    k = DBP_Model(pre)
    index = 10
    error_count = 0
    sen_count=0
    com_count=0
    continu = []
    while index<len(data):
        pre_data = pre[index - 1] + k
        if g>0:
            pre.append(pre_data)
            g -= 1
        else:
            sen_count += 1
            continu.append(Lk)
            if abs(pre_data-data[index]) < thead:#当满足条件时，接着用这次预测值做真实值预测下一次
                Lk += 1
                g = Lk
                pre.append(pre_data)
            else:
                com_count += 1#不满足条件，要传输数据，使用真实值预测下一时刻数据
                Lk = 0
                g = Lk
                pre.append(data[index])
                k = DBP_Model(pre[len(pre) - 10:])
        index += 1


        # pre2.append(pre_data)
        # # print(pre_data)
        # error=abs(pre_data-data[index])
        # # print(pre_data,data[index],error)
        # if error<thead:
        #     pre.append(pre_data)
        # else:
        #     # print("错了",error)
        #     pre.append(data[index])
        #     k=DBP_Model(pre[len(pre)-10:])
        #     error_count+=1
        # index += 1
    # print(error_count)
    return pre,com_count,sen_count

def DBP_withoutskip(data):
    pre=data[0:10]
    pre2=data[0:10]
    thead=0.15
    k=DBP_Model(pre)
    # print(k)
    index=10
    error_count=0

    while index<len(data):
        pre_data = pre[index-1]+k
        pre2.append(pre_data)
        # print(pre_data)
        error=abs(pre_data-data[index])
        # print(pre_data,data[index],error)
        if error<thead:
            pre.append(pre_data)
        else:
            # print("错了",error)
            pre.append(data[index])
            k=DBP_Model(pre[len(pre)-10:])
            error_count+=1
        index += 1
    # print(error_count)
    return pre,pre2,error_count

#线性预测
def func(p,x):
    k,b=p
    return k*x+b
def error(p,x,y):
    return y-func(p,x)

def func2(p,x):
    a,b,r=p
    x1,x2,x3=x
    return a*x1+b*x2+r*x3

def error2(p,x,y):
    return y-func2(p,x)

def linst_model(data):
    pre=np.array(data[0:40])
    x=np.array(list(range(40)))

    p0=[0,0]
    Para=leastsq(error,p0,args=(x,pre))
    k,b0=Para[0]
    # print("k=",k,"b=",b)

    err_tmp=error(Para[0],x,pre)
    m = k * x + b0
    # print(m)

    Y=err_tmp[3:]

    x1=err_tmp[:len(err_tmp)-3]
    x2=err_tmp[1:len(err_tmp)-2]
    x3=err_tmp[2:len(err_tmp)-1]

    X=np.array([x1,x2,x3])
    X1=np.dot(X,X.T)
    Y1=np.dot(X,Y)
    # print(Y1)
    a,b,r=np.linalg.solve(X1, Y1)
    # print(np.linalg.solve(X1, Y1))
    # print(len(X))
    # print(len(Y))
    return X,k,b0,a,b,r

def linst_pr(data):
    pre = data[0:40]
    pre2=data[0:40]
    X,k, b0, a, b, r=linst_model(pre)
    X = X.tolist()
    index=40
    thead=0.15
    i=0
    data_index=index
    count=0
    # print(len(data))

    while data_index<len(data)-3:
        X[0].append(X[1][-1])
        X[1].append(X[2][-1])
        X[2].append(pre[-1]-k*(index-1)-b0)
        # x3.append(pre[j - 3] - k * (j - 3) - b0)
        # x2.append(pre[j - 2] - k * (j - 2) - b0)
        # x1.append(pre[j- 1] - k * (j - 1) - b0)
        y1=k * index + b0
        y2=a * X[0][-1]+b*X[1][-1]+r*X[2][-1]
        pre1 = y1+y2
        error=abs(data[data_index]-pre1)
        pre2.append(pre1)
        # print("pre1=",pre1,"error=",error)
        if error<thead :
            pre.append(pre1)

            index+=1
        else:
            count+=1
            pre.append(data[data_index])

            try:
                X, k, b0, a, b, r = linst_model(pre[len(pre) - 40:])
                X = X.tolist()
                index = 40
            except:
                # print("无解")
                index += 1
        data_index += 1
    # print("count=", count)
    return pre,pre2,count

def draw(data):
    data.plot()
    plt.show()

def writ(data):
    with open('/result.txt','a') as f:
        f.write(data)
        f.close()
# print(pre_tmp)

#求最小支配集，参数:节点：优先级字典({"id":id,"优先级"：优先级}，邻接矩阵
def dominantSet(pri,adj):
    a=set()#判断是否全覆盖
    b={}#优化调整，判断子集的子集是否为支配集
    l=sorted(pri.items(),key=operator.itemgetter(1),reverse=True)
    # print(l)
    p1=[]#按优先级排序的节点号
    p=[]
    for i in range(len(l)):
        p.append(l[i][0])
    # print("结点序号=",p)
    all=set(p)
    result={}
    for i in range(len(p)):#按照优先级遍历每个节点
        indeX=p[i]
        tmp = set()
        tmp.add(indeX)
        for j in range(len(p)):
            if adj[indeX][j]==1:
                tmp.add(j)
        tmpIna=a.intersection(tmp)#tmp与a的交集
        e=tmp.difference(tmpIna)#tmp减去交集后的集合
        if not e:#是空集 pass，不是空集则加入支配集
            pass
        else:
            a.update(tmp)
            # print(a)
            result[indeX]=pri[indeX]#index加入支配集
            b[indeX]=tmp#记录每个支配点支配的节点
        rest=all.difference(a)
        if not rest:#判断是否全覆盖，若是则停止循环
            break
    # print("未更新=",result.keys())
    DM=update(result,b,all)#优化支配集的结果
    EDM={}#被支配集
    EDM_key=list(set(p) ^ set(DM))
    for j in EDM_key:
        dm={}
        for i in range(54):
            if adj[i][j]==1 and i in DM:
                dm[i]=pri[i]
        if len(dm):
            dm=sorted(dm.items(), key=operator.itemgetter(1), reverse=True)
            # print("dm=",dm)
            EDM[j]=dm[0][0]
    # print("EDM=",EDM)
    # print("EDM_key=",EDM_key)
    return DM,b,EDM


def update(result,dic,all):
    a=set()
    m=len(result)
    r = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    l=r
    for i in range(m-1,-1,-1):
        tmp=set()
        d=l[:i]
        d.extend(l[i+1:])
        for i in range(len(d)):
            index = d[i][0]
            tmp.update(dic[index])
        rest=all.difference(tmp)
        if not rest:
            l=d
    id=[]
    for i in range(len(l)):
        id.append(l[i][0])
    print("dic=",dic)
    print("id=",id)
    print("支配点个数：",len(id))
    return id

def corr(data1,data2):#方差的方差作为相关性
    data1 = np.array(data1)
    data2 = np.array(data2)

    diff = data1-data2
    m=np.mean(diff)
    # print(m)
    # m=np.array(m)
    diff2=np.array(np.tile(m,(len(data1))))-diff
    # print(tile(m,(len(data1))))
    sdiff=diff2**2
    sd=sdiff.sum()
    s=sd/(len(data2)-1)
    s=s**0.5
    return s,m

def corre_matrix(data):#计算相关矩阵,data={id:[30个数据]}
    corre = np.zeros((54, 54))  # 相关矩阵
    Mean = np.zeros((54, 54))  # 相差均值
    for i in range(len(data)):
        for j in range(len(data)):
            if i>=j:
                continue
            else:
                # print(id[i],i)
                s,m=corr(data[id[i]],data[id[j]])
                # print(s)
                if s<0.05:
                    corre[id[i]][id[j]] = 1
                    corre[id[j]][id[i]] = 1
                    Mean[id[i]][id[j]] = m#data[id[i]]-data[id[j]]=m
                    Mean[id[j]][id[i]] = -m
    return corre,Mean

def corrdistance(data1,data2):
    data1 = np.array(data1)
    data2 = np.array(data2)

    diff = data1 - data2
    m = np.mean(diff)

if __name__=='__main__':
    #测试并对比预测加不加间隔调整
    # d = getdata()
    # data = {}

    with open('data0228.json', 'r') as f:
        data = json.load(f)
    l1 = list( data.keys())
    # print(l1)

    # d=data["20"]
    #
    # pre1, sen_count1, com_count1 = esmoothpredict_withskip(d)
    # pre_es, com_count2 = esmoothpredict_withoutskip(d)
    # print(len(d),len(pre1),len(pre_es))
    # MSE1 = Mse(pre1, d)
    # MSE2 = Mse(pre_es, d)
    # pre, pre2, count = DBP_withoutskip(d)
    # MSE = Mse(pre, d)
    # pre_Dw, count_Dw ,sen_Dw= DBP_withskip(d)
    # MSE_Dw=Mse(pre_Dw,d)
    #
    # # 线性
    # pre_li, pre2_li, count2 = linst_pr(d)
    # MSE_l = Mse(pre_li, d)
    #
    # print("节点", id)
    # print("MSE_with=", MSE1, "通信数量=", com_count1, "感知数量=", sen_count1)
    # print("MSE_without=", MSE2, "通信数量=", com_count2, "感知数量=", len(d))
    # print("MSE_DBP=", MSE, "通信数量=", count)
    # print("MSE_DBPwith=", MSE_Dw, "通信数量=", count_Dw,"感知数量=", sen_Dw)
    # print("MSE_li=", MSE_l, "通信数量=", count2)

    # c = {"data": d}
    # data2 = pd.DataFrame(c)
    # draw(data2)

    #指数平滑法
    d=[]

    Mse_with=0
    Mse_without=0
    com_with=0
    com_without=0
    sen_with=0
    sen_without=0

    Mse_DBP = 0
    com_DBP = 0
    Mse_DBPw=0
    com_DBPw=0
    sen_DBPw=0

    Mse_li=0
    com_li=0
    for i in range(0,len(data)):
        id = l1[i]
        if id=="20":
            continue
        d=data[id]
        pre1, sen_count1, com_count1 = esmoothpredict_withskip(d)
        pre_es, com_count2 = esmoothpredict_withoutskip(d)
        MSE1=Mse(pre1,d)
        MSE2=Mse(pre_es,d)

        Mse_with+=MSE1
        Mse_without+=MSE2
        com_with+=com_count1
        com_without+=com_count2
        sen_with+=sen_count1
        sen_without+=len(d)
        #DBP
        pre, pre2, count = DBP_withoutskip(d)
        MSE = Mse(pre, d)
        Mse_DBP += MSE
        com_DBP += count

        pre_Dw, count_Dw, sen_Dw = DBP_withskip(d)
        MSE_Dw = Mse(pre_Dw, d)
        Mse_DBPw+=MSE_Dw
        com_DBPw+=count_Dw
        sen_DBPw+=sen_Dw


        #线性
        pre_li,pre2_li,count2=linst_pr(d)
        MSE_l = Mse(pre_li, d)
        Mse_li+=MSE_l
        com_li+=count2
        print("节点",id)
        print( "MSE_with=", MSE1, "通信数量=", com_count1, "感知数量=", sen_count1)
        print("MSE_without=", MSE2, "通信数量=", com_count2, "感知数量=", len(d))
        print("MSE_DBP=", MSE, "通信数量=", count)
        print("MSE_DBPwith=", MSE_Dw, "通信数量=", count_Dw, "感知数量=", sen_Dw)
        print("MSE_li=", MSE_l, "通信数量=", count2)
    print("==============================")
    print("MSE_withoput=", Mse_without, "通信数量=", com_without, "感知数量=", sen_without)
    print("MSE_with=", Mse_with, "通信数量=", com_with, "感知数量=", sen_with)
    print("MSE_li=", Mse_li, "通信数量=", com_li)

    print("MSE_DBP=", Mse_DBP, "通信数量=", com_DBP)
    print("MSE_DBPwith=", Mse_DBPw, "通信数量=", com_DBPw, "感知数量=", sen_DBPw)



    #DBP预测方法
    # Mse_DBP=0
    # com_DBP=0
    # for i in range(0,len(data)):
    #     id = l1[i]
    #     d=data[id]
    #     pre, pre2, count = DBP_withoutskip(d)
    #     MSE = Mse(pre, d)
    #     Mse_DBP += MSE
    #     com_DBP+=count
    # print("MSE_DBP=", Mse_DBP, "通信数量=", com_DBP)
    #
    # pre,pre2,count=DBP_withoutskip(data)
    # print(count)
    # print(pre2)
    #线性预测SAF
    # pre,pre2,count=linst_pr(data)
    # c={"data":data[:len(pre)],"pre":pre}
    # data2=pd.DataFrame(c)
    # draw(data2)
    #########

    #测试dominantSet
    # pri={1:0,2:5,3:1,4:7,5:8,6:9,7:4,8:6,9:3,10:2}
    #
    # l=[[0,1],[1,2],[1,3],[2,3],[2,4],[3,4],[3,5],[3,7],[4,6],[4,7],[4,8],[5,7],[5,9],[6,8],[7,8],[7,9],[8,9]]
    #
    # ad=np.zeros((10,10),int)
    # for k in range(len(l)):
    #     i=l[k][0]
    #     j=l[k][1]
    #     ad[i][j]=1
    #     ad[j][i]=1
    # dominantSet(pri,ad)
    #######

    #数据集54节点空间相似性判断
    # data={}
    # with open('data0228.json','r') as f:
    #     a=json.load(f)
    # corre = np.zeros((54,54))#相关矩阵
    # Mean = np.zeros((54, 54))#相差均值
    # l1=list(a.keys())
    # for i in range(0,len(a)):
    #     id=l1[i]
    #     data[int(id)-1] = a[id][:30]
    #
    # pri = list(data.keys())
    # id = list(data.keys())
    #
    # corre,Mean=corre_matrix(data)

    #算皮尔逊系数
    # d=[]
    # for i in range(54):
    #     for j in range(54):
    #         if corre[i][j]==1 :
    #             print('(%d,%d)' %(i,j))
    #             print(pearsonr(data[i], data[j])[0])
    #             print(Mean[i][j])
    #             d.append([i,j])

    # pri=[21, 15, 0, 42, 2, 47, 5, 10, 44, 18, 48, 24, 40, 32, 31, 30, 11, 43, 17, 26, 33, 7, 16, 19, 51, 9, 50, 38, 8, 23, 3, 14, 35, 45, 25, 28, 22, 34, 6, 36, 13, 20, 52, 46, 39, 53, 12, 1, 37, 29, 49, 41]
    # random.shuffle(pri)
    # print(pri)
    # id_pri = dict(zip(id, pri))
    # DM,dic,EDM=dominantSet(id_pri,corre)
    ########整体网络收集数据,重新计算
    # time=30
    # time_bew=0
    # con_count={}#保存连续叫醒几次
    # for i in range(54):
    #     con_count[i]=0
    # d={}#保存支配集的数据,包括感知和预测
    # for i in DM:
    #     d[i]=esmoothpredict_withskip(a[i][:100])
    # while time<min(len(a)):
    #
    #     time+=1
    #     if time % 100 == 0:  # 重新计算支配集的时间
    #         time_bew = 0
    #         random.shuffle(pri)# 计算优先级pri()
    #         id_pri = dict(zip(id, pri))
    #         for i in range(0,len(a)):
    #             id=l1[i]
    #             data[int(id)-1] = a[id][time:time+30]
    #         corre,Mean=corre_matrix(data)# 计算相关性corre()
    #         DM,dic,EDM=dominantSet(id_pri,corre)# 计算支配集dominantSet(pri,corre)
    #         for i in DM:
    #             d[i] = esmoothpredict_withskip(a[i][time:time+100])
    #     elif time_bew % 5 == 0:  # 没有到重新计算的时间
    #         time_bew+=1
    #         for i in EDM.keys():
    #             if abs(d[EDM[i]][time%100]-a[i][time]-Mean[EDM[i]][i])>0.05:
    #
    #                 if abs(d[EDM[i][time%100+1]-a[i][time+1]-Mean[EDM[i]][i]])>0.05 or abs(d[EDM[i][time%100+2]-a[i][time+2]-Mean[EDM[i]][i]])>0.05:
    #                     DM.append(i)#将该节点加入支配集
    #                     dic[i]=set(i)
    #                 else:




                #真实值-支配集预测值
            #被支配集Ed={id：ed_id}，
                 # 定时唤醒被支配集，可以改成概率控制的
                # 判断是否（真实值-支配集的预测值）超出阈值
                # 若是，则连续检测3个数值，若>=2次都超过阈值，连续监测30次，重新计算相关性，若仍有新的相关性，则按照新的相关关系支配，否则将该点加入支配集
                # 若不是，则讲time_bew的监测间隔加一

        # else:
        #     time_bew += 1
                #用支配集的预测值代替被支配集



    #测试空间相关的两节点并画图
    # data28=[]
    # data28.extend(a['53'][:200])
    # a31 = []
    # a31.extend(a['35'][:200])
    # print(data28)
    # data31 = []
    # data31.extend(data[34])
    # tmp=[]
    # for i in range(30,200):
    #     # print(data28[i])
    #     tmp.append(data28[i]-Mean[52][34])
    # data31.extend(tmp)
    #
    # lenth=min(len(a['36']),len(a['54']))
    # c = {"data1": data28, "data2": data31,"data3": a31}
    # c={"data1": a['36'][:lenth], "data2": a['54'][:lenth]}
    #
    # data = pd.DataFrame(c)
    # data.plot()
    # plt.show()

