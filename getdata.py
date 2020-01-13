import json


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
    data = {}
    i=0
    with open(filename, 'r') as file_to_read:
        lines = file_to_read.readline()
        while lines:
            i += 1
            print(i)
            if len(lines.split()) != 8:
                pass
            # else:
            #     # print(lines.split())
            #     # d_tmp, t_tmp, e_tmp, m_tmp, te_tmp, h_tmp, l_tmp, v_tmp = [i for i in lines.split()]
            #     date = lines.split()[0]
            #     id = int(lines.split()[3])
            #     te = float(lines.split()[4])
            #
            #     if data.__contains__(id):
            #         pass
            #     else:
            #         data[id] = {}
            #     if data[id].__contains__(date):
            #         data[id][date].append(float(te))
            #     else:
            #         data[id][date] = []
            #         data[id][date].append(float(te))
            else:
                date = lines.split()[0]
                id = int(lines.split()[3])
                te = float(lines.split()[4])
                if date!="2004-02-28":
                    pass
                else:
                    if data.__contains__(id):
                        pass
                    else:
                        data[id] = []
                    data[id].append(float(te))
            pass
            lines = file_to_read.readline()

    # data = list(map(float, temperature1))  # temperature1是字符串，将其转换成浮点型数值
    jsondata = json.dumps(data)
    file = open('data0228.json', 'w')
    file.write(jsondata)
    file.close()
    # return data


if __name__ == '__main__':
    # getdata()
    # print(data)
    # data={"1": {"2004-02-28": [19.9884,19.33]}}
    # jsondata = json.dumps(data)
    # file = open('data.json', 'w')
    # file.write(jsondata)
    # file.close()
    l=[1,2,34,5]
    import random
    random.shuffle(l)
    print(l)
