# -*- coding: utf-8 -*-
from configs import *
from datetime import datetime, timedelta
import numpy as np
import cPickle as Pickle
import os.path
from sklearn import preprocessing


def load_data():
    trainx, trainy = read_big(train_traj_path, train_weather_path, mode="train")
    testx, outinfo = read_big(test_traj_path, test_weather_path, mode="test")
    return trainx, trainy, testx, outinfo


# 输入16维 [段编号，段长度，起始时间，天气×6，往后两小时×6，时间段号]
# 输出1维
def read_big(traj, weather, mode):
    with open(traj, "rb") as f:
        f.readline()
        traj_data = f.readlines()
    with open(weather, "rb") as f:
        f.readline()
        weather_data = f.readlines()

    travel_times = {}  # 每一段路径，每一段时间中的travel time
    padding = {}  # 每一段路径，每一段时间，所有天平均(中位数)travel time，用于填充空白值

    # "B","3","1065642","2016-07-19 00:14:24",
    # "105#2016-07-19 00:14:24#9.56;100#2016-07-19 00:14:34#6.75;111#2016-07-19 00:14:41#13.00;103#2016-07-19 00:14:54#7.47;122#2016-07-19 00:15:02#32.85",
    # "70.85"

    for i, this_traj in enumerate(traj_data):
        arr = this_traj.replace('"', '').split(",")

        travel_time = float(arr[5])

        # 删除异常值
        if arr[0] == "B":
            if arr[1] == "3":
                if travel_time > 250:
                    continue
            else:
                if travel_time > 250 or travel_time < 40:
                    continue

        elif arr[0] == "A":
            if arr[1] == "3":
                if travel_time > 350:
                    continue
            else:
                if travel_time > 200:
                    continue
        else:
            if arr[1] == "3":
                if travel_time > 400:
                    continue
            else:
                if travel_time > 400 or travel_time < 70:
                    continue

        route_name = " ".join([arr[0], arr[1]])
        trace_start_time = datetime.strptime(arr[3], "%Y-%m-%d %H:%M:%S")
        time_window_minute = trace_start_time.minute // 20 * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)

        if route_name not in travel_times:
            travel_times[route_name] = {start_time_window: [travel_time]}
        else:
            if start_time_window in travel_times[route_name]:
                travel_times[route_name][start_time_window].append(travel_time)
            else:
                travel_times[route_name][start_time_window] = [travel_time]

        day_window = datetime(2016, 1, 1, trace_start_time.hour, time_window_minute, 0)

        if route_name not in padding:
            padding[route_name] = {day_window: [travel_time]}
        else:
            if day_window in padding[route_name]:
                padding[route_name][day_window].append(travel_time)
            else:
                padding[route_name][day_window] = [travel_time]

    for route_name in travel_times:
        for travel_time in travel_times[route_name]:
            travel_times[route_name][travel_time] = np.mean(travel_times[route_name][travel_time])
        for day_window in padding[route_name]:
            padding[route_name][day_window] = np.median(padding[route_name][day_window])

    weather = {}  # 每时段的天气特征
    # "2016-07-01","0","1000.4000","1005.3000","225.0000","2.1000","26.4000","94.0000","0.0000"
    for i, this_weather in enumerate(weather_data):
        arr = this_weather.replace('"', '').split(',')

        date_day = datetime.strptime(arr[0], '%Y-%m-%d')
        hour = arr[1]
        date_day = datetime(date_day.year, date_day.month, date_day.day, int(hour))
        weather[date_day] = arr[2:8]

    if mode == "train":
        x = []
        y = []
        for route_name in travel_times:
            for travel_time in travel_times[route_name]:
                if travel_time.hour >= 6 and travel_time.hour <= 18:
                    thisx = [convert_route(route_name), route_len(route_name), travel_time.hour * 60. + travel_time.minute]
                    weather_time = datetime(travel_time.year, travel_time.month, travel_time.day,
                                            (travel_time.hour + 2) // 3 * 3)
                    thisx = thisx + weather[weather_time]

                    for n in range(6):
                        time = travel_time + timedelta(minutes=20 * n)
                        padding_time = datetime(2016, 1, 1, time.hour, time.minute, 0)
                        thisx.append(travel_times[route_name][time] if time in travel_times[route_name] else padding[route_name][padding_time])

                    for n in range(6):
                        x.append(thisx + [n])

                        time = travel_time + timedelta(minutes=20 * (n+6))
                        padding_time = datetime(2016, 1, 1, time.hour, time.minute, 0)
                        if time in travel_times[route_name]:
                            thisy = travel_times[route_name][time]
                        else:
                            thisy = padding[route_name][padding_time]
                        y.append(thisy)

        return updatex(x), np.array(y, dtype=float)

    else:
        x = []
        outinfo = []
        test_times = []
        for i in range(7):
            test_times.append(datetime(2016, 10, 18, 6, 0, 0))
            test_times.append(datetime(2016, 10, 18, 15, 0, 0))
        for route_name in travel_times:
            for travel_time in test_times:
                if (travel_time.hour == 6 or travel_time.hour == 15) and int(travel_time.minute) == 0:
                    thisx = [convert_route(route_name), route_len(route_name), travel_time.hour * 60. + travel_time.minute]
                    weather_time = datetime(travel_time.year, travel_time.month, travel_time.day,
                                            (travel_time.hour + 2) // 3 * 3)
                    thisx = thisx + weather[weather_time]

                    for n in range(6):
                        time = travel_time + timedelta(minutes=20 * n)
                        padding_time = datetime(2016, 1, 1, time.hour, time.minute, 0)
                        thisx.append(
                            travel_times[route_name][time] if time in travel_times[route_name] else padding[route_name][
                                padding_time])

                    for n in range(6):
                        x.append(thisx + [n])
                        outtime = travel_time + timedelta(minutes=20 * (n+6))
                        outinfo.append([route_name, outtime])

        return updatex(x), outinfo


def convert_route(route):
    if route == "B 3":
        return 0.
    elif route == "B 1":
        return 1.
    elif route == "A 2":
        return 2.
    elif route == "A 3":
        return 3.
    elif route == "C 1":
        return 4.
    else:
        return 5.


def updatex(x):
    updatex = np.array(x, dtype=float)
    return updatex


# 每一段长度,通过routes文件手工计算
def route_len(route):
    if route == "B 3":
        return 477.
    elif route == "B 1":
        return 821.
    elif route == "A 2":
        return 384.
    elif route == "A 3":
        return 852.
    elif route == "C 1":
        return 1678.
    else:
        return 1252.


def mscale(train_x, train_y, test_x):
    scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scalerY = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_x = scalerX.fit_transform(train_x)
    train_y = scalerY.fit_transform(train_y)
    test_x = scalerX.transform(test_x)
    return train_x, train_y, test_x, scalerX, scalerY


def write_prediction(predy, outinfo):
    with open(save_data_path + "Travel_Time_Prediction_big.csv", 'wb') as fw:
        fw.write(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
        for i, item in enumerate(outinfo):
            route_name, outtime = item
            write_arr = ['"' + x + '"' for x in route_name.split(" ")]
            write_arr.append(
                '"[' + ",".join([str(outtime), str(outtime + timedelta(minutes=20))]) + ')"')
            write_arr.append('"' + str(predy[i]) + '"\n')
            fw.write(",".join(write_arr))


if __name__ == "__main__":
    x, y = read_big(train_traj_path, train_weather_path, "train")
    print x.shape, y.shape
    print x[0:10]