# -*- coding:utf-8 -*-
import numpy
import csv
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

def in_file(file_name):
    csvfile = file(file_name,'rb')
    reader = csv.reader(csvfile)
    sub = []
    for line in reader:
        sub.append(line)
    sub.pop(0)

    out = []
    for i in range(len(sub)):
        if sub[i][0] == 'A' and sub[i][1] == '2':
            out.append(sub[i])
    for i in range(len(sub)):
        if sub[i][0] == 'A' and sub[i][1] == '3':
            out.append(sub[i])
    for i in range(len(sub)):
        if sub[i][0] == 'B' and sub[i][1] == '1':
            out.append(sub[i])
    for i in range(len(sub)):
        if sub[i][0] == 'B' and sub[i][1] == '3':
            out.append(sub[i])
    for i in range(len(sub)):
        if sub[i][0] == 'C' and sub[i][1] == '1':
            out.append(sub[i])
    for i in range(len(sub)):
        if sub[i][0] == 'C' and sub[i][1] == '3':
            out.append(sub[i])

    data1 = []
    data2 = []
    for i in range(len(out)):
        if 0<= i % 12 <=5:
            data1.append(out[i])
        else:
            data2.append(out[i])
    data_all = data1 + data2
    return data_all

def add_or_multiple(data_all):
    for i in range(0,len(data_all)):
        tmp = float(data_all[i][3])
        #tmp = float(data_all[i][3])*1.1
        data_all[i][3] = float('%.2f' % tmp)
    return data_all

def MAPE_and_plot(data_all):
    csvfile = file('test_offline.csv','rb')
    reader = csv.reader(csvfile)
    true_value = []
    for line in reader:
        true_value.append(float(line[3]))

    sub = []
    for i in range(len(data_all)):
        sub.append(float(data_all[i][3]))

    l = 0
    count = 0
    for i in range(len(sub)):
        if true_value[i] != 0:
            tmp = abs(sub[i] - true_value[i])/float(true_value[i])
            l = l + tmp
            count = count + 1
    MAPE = l/float(count)
    print MAPE

    plt.plot(true_value)
    plt.plot(sub)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    data_all = in_file('Travel_Time_Prediction.csv')
    data_all = add_or_multiple(data_all)
    MAPE_and_plot(data_all)


