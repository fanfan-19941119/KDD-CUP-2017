# -*- coding:utf-8 -*-
import numpy
import csv
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from configs import *

def ssj_file(file_name):
    csvfile = file(file_name,'rb')
    reader = csv.reader(csvfile)
    sub = []
    for line in reader:
        sub.append(line[3])
    sub.pop(0)
    for i in range(len(sub)):
        sub[i] = float(sub[i])
    ssj = sub
    return ssj

def dz_xgb_file(file_name):
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
    
    for i in range(len(data_all)):
        tmp = float(data_all[i][3])+10
        # tmp = float(data_all[i][3])
        data_all[i][3] = float('%.2f' % tmp)
    
    dz = []
    for i in range(len(data_all)):
        dz.append(float(data_all[i][3]))
    return dz

def dz_nn_file(file_name):
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
    
    dz = []
    for i in range(len(data_all)):
        dz.append(float(data_all[i][3]))
    return dz

def merge(merge_list):
    merge = []
    len_merge_list = float(len(merge_list))
    for i in range(len(merge_list[0])):
        tmp = 0
        for j in range(len(merge_list)):
            tmp = tmp + merge_list[j][i]
        tmp = tmp/len_merge_list
        merge.append(tmp)
    return merge

def MAPE(value):
    csvfile = file('dataSets/test_offline.csv','rb')
    reader = csv.reader(csvfile)
    true_value = []
    for line in reader:
        true_value.append(float(line[3]))

    l = 0
    count = 0
    for i in range(len(value)):
        if true_value[i] != 0:
            tmp = abs(value[i] - true_value[i])/float(true_value[i])
            l = l + tmp
            count = count + 1
    MAPE = l/float(count)
    print MAPE

    plt.plot(true_value)
    plt.plot(value)
    plt.grid(True)
    # plt.show()
    
if __name__ == '__main__':
    ssj_file_name = []
    #ssj_file_name = []
    ssj_list = []
    for i in range(len(ssj_file_name)):
        tmp = ssj_file(ssj_file_name[i])
        ssj_list.append(tmp)

    #dz_xgb_file_name = ['F:\\KDD CUP 2017\\2017_5_24\\8_4_MAPE0.19909.csv']
    # dz_xgb_file_name = [save_model_path + 'Travel_Time_Prediction.csv']
    dz_xgb_file_name = ["/media/workserv/Seagate Backup Plus Drive/tianchi/dataPredict/2017_5_24/8_3_MAPE0.19907.csv"]
    # dz_xgb_file_name = []
    dz_xgb_list = []
    for i in range(len(dz_xgb_file_name)):
        tmp = dz_xgb_file(dz_xgb_file_name[i])
        dz_xgb_list.append(tmp)

    dz_nn_file_name = ["/media/workserv/Seagate Backup Plus Drive/tianchi/dataPredict/2017_5_17/Travel_Time_Prediction.csv"]
    dz_nn_list = []
    for i in range(len(dz_nn_file_name)):
        tmp = dz_nn_file(dz_nn_file_name[i])
        dz_nn_list.append(tmp)

    merge_list = ssj_list + dz_xgb_list + dz_nn_list
    merge = merge(merge_list)

    for i in range(len(ssj_list)):
        MAPE(ssj_list[i])
    for i in range(len(dz_xgb_list)):
        MAPE(dz_xgb_list[i])
    for i in range(len(dz_nn_list)):
        MAPE(dz_nn_list[i])

    print 'merge_MAPE:'
    MAPE(merge)
