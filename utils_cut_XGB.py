# -*- coding: utf-8 -*-
from configs import *
from datetime import datetime, timedelta
import numpy as np
import cPickle as Pickle
import os.path
from sklearn import preprocessing

np.random.seed(1337)

def load_data(offline=False):
    flag = os.path.exists(online_data_path) and os.path.exists(offline_data_path) \
           and os.path.exists(routes_data_path)

    if not flag:
        (train_x, train_y) = prePro_Time(train_traj_path, train_weather_path, 'train')
        (test_x, test_y, actual_time, date) = prePro_Time(test_traj_path, test_weather_path, 'test')
        Pickle.dump((train_x, train_y, test_x, test_y, actual_time, date), open(online_data_path, "wb"), protocol=True)

        (train_x, train_y) = prePro_Time(train_traj_path, train_weather_path, "offline_train")
        (test_x, test_y, actual_time, date) = prePro_Time(train_traj_path, train_weather_path, "offline_test")
        Pickle.dump((train_x, train_y, test_x, test_y, actual_time, date), open(offline_data_path, "wb"), protocol=True)

        Pickle.dump(get_routes(), open(routes_data_path, "wb"), protocol=True)

    this_p = offline_data_path if offline else online_data_path

    (train_x, train_y, test_x, test_y, actual_time, date) = Pickle.load(open(this_p, "rb"))
    routes = Pickle.load(open(routes_data_path, "rb"))

    return (train_x, train_y, test_x, test_y, actual_time, date, routes)


# 输入17维 [段编号，道路长，宽，起始时间，往后两小时×6，天气×7]
# 输出6维
def prePro_Time(traj, weather, mode):
    fr1 = open(traj, 'r')
    fr1.readline()  # skip the header
    traj_data = fr1.readlines()
    fr1.close()

    fr2 = open(weather, 'r')
    fr2.readline()  # skip the header
    weather_data = fr2.readlines()
    fr2.close()

    fr3 = open(linkes_path, 'r')
    fr3.readline()  # skip the header
    links_data = fr3.readlines()
    fr3.close()

    travel_times = {}  # 每一段路径，每一段时间中的travel time
    day_average = {}  # 每一段路径，每一段时间，所有天平均travel time，用于填充空白值
    travel_times_actual = {}
    # "B","3","1065642","2016-07-19 00:14:24",
    # "105#2016-07-19 00:14:24#9.56;100#2016-07-19 00:14:34#6.75;111#2016-07-19 00:14:41#13.00;103#2016-07-19 00:14:54#7.47;122#2016-07-19 00:15:02#32.85",
    # "70.85"

    for i, this_traj in enumerate(traj_data):
        arr = this_traj.replace('"', '').split(",")

        travel_time = float(arr[5])
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

        trace_time = arr[4].split(';')

        for section in trace_time:
            section = section.split('#')
            if section[0] not in travel_times:
                travel_times[section[0]] = {}
            if section[0] not in day_average:
                day_average[section[0]] = {}

            trace_start_time = datetime.strptime(section[1], "%Y-%m-%d %H:%M:%S")
            time_window_minute = trace_start_time.minute // 20 * 20
            start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                         trace_start_time.hour, time_window_minute, 0)

            tt = float(section[2])  # travel time
            if start_time_window not in travel_times[section[0]]:
                travel_times[section[0]][start_time_window] = [tt]
            else:
                travel_times[section[0]][start_time_window].append(tt)

            day_window = datetime(2016, 7, 19, trace_start_time.hour, time_window_minute, 0)
            if day_window not in day_average[section[0]]:
                day_average[section[0]][day_window] = [tt]
            else:
                day_average[section[0]][day_window].append(tt)

        route = " ".join([arr[0], arr[1]])
        if route not in travel_times_actual:
            travel_times_actual[route] = {}
            travel_times_actual[route][start_time_window] = [float(arr[5])]
        else:
            if start_time_window not in travel_times_actual[route]:
                travel_times_actual[route][start_time_window] = [float(arr[5])]
            else:
                travel_times_actual[route][start_time_window].append(float(arr[5]))

    weather = {}  # 每时段的天气特征
    # "2016-07-01","0","1000.4000","1005.3000","225.0000","2.1000","26.4000","94.0000","0.0000"
    for i, this_weather in enumerate(weather_data):
        arr = this_weather.replace('"', '').split(',')

        date_day = datetime.strptime(arr[0], '%Y-%m-%d')
        hour = arr[1]
        date_day = datetime(date_day.year, date_day.month, date_day.day, int(hour))
        weather[date_day] = arr[2:9]

    links = {}
    #"link_id","length","width","lanes","in_top","out_top","lane_width"
    #"100","58","3","1","105","111","3"
    for this_link in links_data:
        arr = this_link.replace('"', '').split(',')
        links[arr[0]] = arr[1:3]

    link_all = list(travel_times.keys())
    link_all.sort()

    for link_id in link_all:
        route_time_windows = list(travel_times[link_id].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            travel_times[link_id][time_window_start] = np.mean(travel_times[link_id][time_window_start])

        day_window = list(day_average[link_id].keys())
        day_window.sort()
        for day_window_start in day_window:
            day_average[link_id][day_window_start] = np.median(day_average[link_id][day_window_start])

    x = []
    if mode == "train" or mode == "offline_train":
        y = []
        for link_id in link_all:
            for time_window_start in route_time_windows:
                this_x = [int(link_id), int(links[link_id][0]), int(links[link_id][1]),
                          time_window_start.hour * 60 + time_window_start.minute]
                this_y = []
                if mode == "offline_train":
                    flag = (time_window_start < datetime(2016, 10, 11))
                else:
                    flag = True
                if flag and time_window_start.hour >= 6 and time_window_start.hour < 18:
                    for n in range(6):
                        time = time_window_start + timedelta(minutes=20 * n)
                        day_time = datetime(2016, 7, 19, time.hour, time.minute, 0)
                        this_x.append(travel_times[link_id][time] if time in travel_times[link_id] else day_average[link_id][day_time])

                        time = time_window_start + timedelta(minutes=20 * (n + 6))
                        day_time = datetime(2016, 7, 19, time.hour, time.minute, 0)
                        this_y.append(travel_times[link_id][time] if time in travel_times[link_id] else day_average[link_id][day_time])

                    time = datetime(time_window_start.year, time_window_start.month, time_window_start.day,
                                    (time_window_start.hour+2) // 3 * 3)

                    for e in weather[time]:
                        this_x.append(float(e))

                    x.append(this_x)
                    y.append(this_y)

        return updatexy(x, y)

    else:
        y = []
        test_date = {}
        # actual_time = {}
        count = 0
        for link_id in link_all:
            for time_window_start in route_time_windows:
                this_x = [int(link_id), int(links[link_id][0]), int(links[link_id][1]),
                          time_window_start.hour * 60 + time_window_start.minute]
                this_y = []
                flag = (time_window_start >= datetime(2016, 10, 11)) \
                       and (time_window_start.hour == 6 or time_window_start.hour == 15) \
                       and int(time_window_start.minute) == 0
                if flag:
                    for n in range(6):
                        time = time_window_start + timedelta(minutes=20 * n)

                        day_time = datetime(2016, 7, 19, time.hour, time.minute, 0)
                        this_x.append(travel_times[link_id][time] if time in travel_times[link_id] else day_average[link_id][day_time])

                        if mode == "offline_test":
                            time = time_window_start + timedelta(minutes=20 * (n + 6))
                            this_y.append(travel_times[link_id][time] if time in travel_times[link_id] else day_average[link_id][day_time])

                        else:
                            this_y.append(travel_times[link_id][time] if time in travel_times[link_id] else 0)

                    time = datetime(time_window_start.year, time_window_start.month, time_window_start.day,
                                    (time_window_start.hour+2) // 3 * 3)

                    for e in weather[time]:
                        this_x.append(float(e))

                    x.append(this_x)
                    y.append(this_y)

                    if time_window_start not in test_date:
                        test_date[time_window_start] = {}
                    test_date[time_window_start][link_id] = count
                    count += 1

        x, y = updatexy(x, y)
        return x, y, travel_times_actual, test_date


def updatexy(x, y):
    x = np.array(x)
    updatex = np.zeros((6 * x.shape[0], x.shape[1] + 1))
    if y != []:
        y = np.array(y)
        updatey = np.zeros((6 * y.shape[0]))
    l = x.shape[0]
    for i in range(6):
        updatex[i * l:(i + 1) * l, 0] = np.ones((l)) * i
        updatex[i * l:(i + 1) * l, 1:(x.shape[1] + 1)] = x
        if y != []:
            updatey[i * l:(i + 1) * l] = y[:, i]

    updatex = np.delete(updatex, [17], axis=1)
    updatex = np.column_stack((updatex, np.square(updatex[:, 10])))
    if y == []:
        return updatex
    else:
        return updatex, updatey


def mscale(train_x, train_y, test_x):
    scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scalerY = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_x = scalerX.fit_transform(train_x)
    train_y = scalerY.fit_transform(train_y)
    test_x = scalerX.transform(test_x)
    return train_x, test_x, train_y, scalerX, scalerY


def get_routes():
    routes = {}
    with open(route_path, 'r') as fr_routes:
        fr_routes.readline()  # skip the header
        for line in fr_routes:
            arr = line.strip().replace('"', '').split(",")
            r = " ".join([arr[0], arr[1]])
            routes[r] = []
            for num in arr[2:]:
                routes[r].append(num)
    return routes



def get_MAPE(date, routes, predict_y, actual_time, test_y):
    MAPE = 0
    m=0
    count = 0.
    not_exist = 0
    l = list(date.keys())
    l.sort()
    for start_time in l:
        for delta in range(6):
            for link in routes:
                predict_time = 0
                this_actual_time = 0
                ttt = 0
                for ilink in routes[link]:
                    predict_time += abs(predict_y[date[start_time][ilink]][delta])
                    ttt += test_y[date[start_time][ilink]][delta]
                time = start_time+timedelta(hours=2, minutes=20*delta)
                if time in actual_time[link]:
                    this_actual_time += np.mean(actual_time[link][time])
                else:
                    this_actual_time = predict_time
                    not_exist += 1

                MAPE += abs(predict_time - this_actual_time) / this_actual_time

                m += abs(predict_time - ttt) /ttt
                count += 1

    MAPE = MAPE / (count - not_exist)
    m = m/count
    print count, not_exist
    # print MAPE - m
    # print m
    print("MAPE = %0.5f, m = %0.5f\n"%(MAPE, m))
    return MAPE


def write_prediction(date, routes, predict_y):
    with open(save_data_path+"Travel_Time_Prediction_cut_XGB.csv", 'wb') as fw:
        fw.write(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
        l = list(date.keys())
        l.sort()
        for start_time in l:
            for delta in range(6):
                for link in routes:
                    write_arr = ['"' + x + '"' for x in link.split(" ")]
                    this_start_time = start_time + timedelta(minutes=20 * (delta + 6))
                    write_arr.append(
                        '"[' + ",".join([str(this_start_time), str(this_start_time + timedelta(minutes=20))]) + ')"')
                    time = 0
                    for ilink in routes[link]:
                        time += predict_y[date[start_time][ilink]][delta]
                    write_arr.append('"' + str(time + 10) + '"\n')
                    # +10为测试得到的后处理
                    fw.write(",".join(write_arr))