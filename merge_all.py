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


def dz_file(file_name):
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


def merge(merge_list, weight_list):
    merge = []
    for i in range(len(merge_list[0])):
        tmp = 0
        for j in range(len(merge_list)):
            tmp = tmp + weight_list[j] * merge_list[j][i]
        merge.append(tmp)
    return merge


# def MAPE(value):
#     csvfile = file('dataSets/test_offline.csv','rb')
#     reader = csv.reader(csvfile)
#     true_value = []
#     for line in reader:
#         true_value.append(float(line[3]))
#
#     l = 0
#     count = 0
#     for i in range(len(value)):
#         if true_value[i] != 0:
#             tmp = abs(value[i] - true_value[i])/float(true_value[i])
#             l = l + tmp
#             count = count + 1
#     MAPE = l/float(count)
#     print MAPE
#
#     plt.plot(true_value)
#     plt.plot(value)
#     plt.grid(True)
#     plt.show()
    
if __name__ == '__main__':
    ssj_file_name = []
    ssj_list = []
    for i in range(len(ssj_file_name)):
        tmp = ssj_file(ssj_file_name[i])
        ssj_list.append(tmp)

    dz_file_name = ["dataPredict/Travel_Time_Prediction_big.csv"]
    dz_list = []
    for i in range(len(dz_file_name)):
        tmp = dz_file(dz_file_name[i])
        dz_list.append(tmp)

    merge_list = ssj_list + dz_list
    weight_list = [1]
    merge = merge(merge_list, weight_list)

    # for i in range(len(ssj_list)):
    #     MAPE(ssj_list[i])
    # for i in range(len(dz_list)):
    #     MAPE(dz_list[i])
    #
    # print 'merge_MAPE:'
    # MAPE(merge)

    csvfile = file('dataSets/submission_sample_travelTime.csv', 'rb')
    reader = csv.reader(csvfile)
    sample = []
    for line in reader:
        sample.append(line)
    for i in range(1, len(sample)):
        sample[i][3] = merge[i - 1]

    csvfile = file('dataPredict/submission_travelTime_merge_final.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(sample)
    csvfile.close()
