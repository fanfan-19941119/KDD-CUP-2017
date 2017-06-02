# -*- coding: utf-8 -*-
from configs import *
from datetime import datetime, timedelta
import numpy as np
import pickle as Pickle
import os.path


def load_data(offline=False):
    flag = os.path.exists(online_data_path) and os.path.exists(offline_data_path) \
           and os.path.exists(routes_data_path)

    if not flag:
        (train_x, train_y) = prePro_Time(train_traj_path, train_weather_path, 'train')
        (test_x, test_y, date) = prePro_Time(test_traj_path, test_weather_path, 'test')
        Pickle.dump((train_x, train_y, test_x, test_y, date), open(online_data_path, "wb"), protocol=True)

        (train_x, train_y) = prePro_Time(train_traj_path, train_weather_path, "offline_train")
        (test_x, test_y, date) = prePro_Time(train_traj_path, train_weather_path, "offline_test")
        Pickle.dump((train_x, train_y, test_x, test_y, date), open(offline_data_path, "wb"), protocol=True)

        Pickle.dump(get_routes(), open(routes_data_path, "wb"), protocol=True)

    this_p = offline_data_path if offline else online_data_path

    (train_x, train_y, test_x, test_y, date) = Pickle.load(open(this_p, "rb"))
    routes = Pickle.load(open(routes_data_path, "rb"))

    return (train_x, train_y, test_x, test_y, date, routes)


# 输入15维 [段编号，起始时间，往后两小时×6，天气×7]
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
    # "B","3","1065642","2016-07-19 00:14:24",
    # "105#2016-07-19 00:14:24#9.56;100#2016-07-19 00:14:34#6.75;111#2016-07-19 00:14:41#13.00;103#2016-07-19 00:14:54#7.47;122#2016-07-19 00:15:02#32.85",
    # "70.85"

    for i, this_traj in enumerate(traj_data):
        arr = this_traj.replace('"', '').split(",")
        travel_time=float(arr[5])
        if travel_time>500:
            continue
        trace_time = arr[4].split(';')

        for section in trace_time:
            section = section.split('#')
            if section[0] not in travel_times:
                travel_times[section[0]] = {}
            trace_start_time = datetime.strptime(section[1], "%Y-%m-%d %H:%M:%S")
            time_window_minute = trace_start_time.minute//20 * 20
            start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                         trace_start_time.hour, time_window_minute, 0)

            tt = float(section[2])  # travel time
            if start_time_window not in travel_times[section[0]]:
                travel_times[section[0]][start_time_window] = [tt]
            else:
                travel_times[section[0]][start_time_window].append(tt)

    weather = {}  # 每时段的天气特征
    # "2016-07-01","0","1000.4000","1005.3000","225.0000","2.1000","26.4000","94.0000","0.0000"

    for i, this_weather in enumerate(weather_data):
        arr = this_weather.replace('"', '').split(',')

        date_day = datetime.strptime(arr[0], '%Y-%m-%d')
        hour = arr[1]
        date_day = datetime(date_day.year, date_day.month, date_day.day, int(hour))
        weather[date_day] = arr[2:9]
    
    links={} #feature for every links
    #"link_id","length","width","lanes","in_top","out_top","lane_width"
    #"100","58","3","1","105","111","3"
    for i, this_weather in enumerate(links_data):
        arr = this_weather.replace('"', '').split(',')
        links[arr[0]] = arr[1:3]

    link_all = list(travel_times.keys())
    link_all.sort()

    for link_id in link_all:
        route_time_windows = list(travel_times[link_id].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            travel_times[link_id][time_window_start] = np.mean(travel_times[link_id][time_window_start])

    x = []
    if mode == "train" or mode == "offline_train":
        y = []
        for link_id in link_all:
            for time_window_start in route_time_windows:
                this_x = [int(link_id),int(links[link_id][0]),int(links[link_id][1]),\
                          time_window_start.hour * 60 + time_window_start.minute]
                this_y = []
                if mode == "offline_train":
                    flag = (time_window_start < datetime(2016, 10, 11))
                else:
                    flag = True
                if flag and time_window_start.hour >= 5 and time_window_start.hour < 20:
                    for n in range(6):
                        time = time_window_start + timedelta(minutes=20 * n)
                        this_x.append(travel_times[link_id][time] if time in travel_times[link_id] else 0)
                        time = time_window_start + timedelta(minutes=20 * (n + 6))
                        this_y.append(travel_times[link_id][time] if time in travel_times[link_id] else 0)

                    time = datetime(time_window_start.year, time_window_start.month, time_window_start.day,
                                     time_window_start.hour // 3 * 3)
                    if time in weather:
                         for e in weather[time]:
                             this_x.append(float(e))
                    else:
                         this_x += [0 for _ in range(7)]
                    x.append(this_x)
                    y.append(this_y)

        return normalization(x, y)

    else:
        y = []
        test_date = {}
        count = 0
        for link_id in link_all:
            for time_window_start in route_time_windows:
                this_x = [int(link_id), int(links[link_id][0]),int(links[link_id][1]),\
                          time_window_start.hour * 60 + time_window_start.minute]
                this_y = []
                flag = (time_window_start >= datetime(2016, 10, 11)) \
                       and (time_window_start.hour == 6 or time_window_start.hour == 15) \
                       and int(time_window_start.minute) == 0
                if flag:
                    for n in range(6):
                        time = time_window_start + timedelta(minutes=20 * n)
                        this_x.append(travel_times[link_id][time] if time in travel_times[link_id] else 0)
                        time = time_window_start + timedelta(minutes=20 * (n + 6))
                        this_y.append(travel_times[link_id][time] if time in travel_times[link_id] else 0)

                    time = datetime(time_window_start.year, time_window_start.month, time_window_start.day,
                                    time_window_start.hour//3 * 3)
                    if time in weather:
                         for e in weather[time]:
                             this_x.append(float(e))
                    else:
                         this_x += [0 for _ in range(7)]
                    x.append(this_x)
                    y.append(this_y)

                    if time_window_start not in test_date:
                        test_date[time_window_start] = {}
                    test_date[time_window_start][link_id] = count
                    count += 1

        x, y = normalization(x, y)
        return x, y, test_date


def normalization(x, y):
    for i in range(len(x)):
        x[i][0] -= 100.
        x[i][1] /= 200
        x[i][2] /= 12
        x[i][3] /= 500
        for n in range(6):
            x[i][n+4] /= 10
        x[i][10] = (x[i][9] - 990) / 10
        x[i][11] = (x[i][10] - 1000) / 10
        x[i][12] = 1 if x[i][11] > 90000 else x[i][11] / 360
        x[i][13] /= 5
        x[i][14] /= 30
        x[i][15] /= 50
    for i in range(len(y)):
        for n in range(6):
            y[i][n] /= 10
    if y == []:
        return x
    else:
        return  x, y


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


def get_MAPE(date, routes, predict_y, test_y):
    MAPE = 0
    num = 0
    l = list(date.keys())
    l.sort()
    for start_time in l:
        for delta in range(6):
            for link in routes:
                predict_time = 0
                actual_time = 0
                for ilink in routes[link]:
                    predict_time += abs(predict_y[date[start_time][ilink]][delta])
                    actual_time += test_y[date[start_time][ilink]][delta]
                if actual_time!=0 :
                    MAPE +=abs(predict_time - actual_time) / actual_time
                    num=num+1

    MAPE = MAPE / num
    print("MAPE = %0.5f\n"%MAPE)
    return MAPE


def write_prediction(date, routes, predict_y):
    with open(save_task1_path, 'w') as fw:
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
                        time += predict_y[date[start_time][ilink]][delta] * 10
                    write_arr.append('"' + str(time) + '"\n')
                    fw.write(",".join(write_arr))
