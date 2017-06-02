# coding=utf-8

import numpy as np
import xgboost as xgb
import utils_big

# from configs import save_model_path

# 输入16维 [段编号，段长度，起始时间，天气×6，往后两小时×6，时间编号]
# [0,       1,          2,          3,              4,              5,              6,          7,              8,              9,  10, 11, 12, 13, 14, 15]
# [routeid, starttime,  routelen,   pressure,       sea_preassure,  wind_direction, wind_speed, temperature,    rel_humidity,   t0, t1, t2, t3, t4, t5, time_id]
# [pressure, sea_preassure, wind_direction, wind_speed, temperature, rel_humidity, precipitation]

CV = False

(train_x, train_y, test_x, outinfo) = utils_big.load_data()
print(len(train_x), len(train_y), len(test_x))
print(train_x.shape)

#记录程序运行时间
import time
start_time = time.time()

#读入数据
params={
    'booster':'gbtree',
    'objective':'reg:logistic', #逻辑回归问题
    'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':8, # 构建树的深度，越大越容易过拟合
    # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.9, # 随机采样训练样本
    'colsample_bytree':0.9, # 生成树时进行的列采样
    'min_child_weight':4,
    'eta': 0.01, # 如同学习率
    'seed':1000,
    "silent": 1
    }


plst = list(params.items())
num_rounds = 10000 # 迭代次数

trainx, trainy, testx, scalerX, scalerY = utils_big.mscale(train_x, train_y, test_x)

xgb_train = xgb.DMatrix(trainx, label=trainy)

xgb_test = xgb.DMatrix(testx)


def mapeobj(preds, dtrain):
    gaps = dtrain.get_label()
    grad = np.sign(preds-gaps)/gaps
    hess = 1/gaps
    grad[(gaps==0)] = 0
    hess[(gaps==0)] = 0
    return grad,hess


def evalmape(preds, dtrain):
    gaps = dtrain.get_label()
    err = abs(gaps-preds)/gaps
    err[(gaps==0)] = 0
    err = np.mean(err)
    return 'error',err

if CV:
    # for sss in [0.7, 0.8, 0.9]:
    #     for ccc in [0.7, 0.8, 0.9]:
    #         params["subsample"] = sss
    #         params["colsample_bytree"] = ccc
    # for gg in [0.05, 0.1, 0.15, 0.2]:
    #     params["gamma"] = gg
    # for ddd in range(7, 10, 1):
    #     for mmm in range(2, 5, 1):
    #         params["min_child_weight"] = mmm
    #         params["max_depth"] = ddd
    plst = list(params.items())
    print plst
    model = xgb.cv(plst, xgb_train, num_rounds, nfold=5, early_stopping_rounds=50, obj=mapeobj, feval=evalmape)
    # print('max_depth = %d, min_child_weight = %d' % (ddd, mmm))
        # print('subsample = %f, colsample_bytree = %f' % (sss, ccc))
    # print('gamma = %f' % (gg))
    print model.values[model.values.argmin(0)[0]][0], model.values.argmin(0)[0]
    print "\n"

else:
    model = xgb.train(plst, xgb_train, 19, obj=mapeobj, feval=evalmape)
    # model.save_model(save_model_path +"gamma02eta001.model") # 用于存储训练出的模型
    # print("best best_ntree_limit",model.best_ntree_limit)

    preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)

    predict_y = scalerY.inverse_transform(preds)
    utils_big.write_prediction(predict_y, outinfo)

# #输出运行时长
cost_time = time.time()-start_time
print "xgboost success!\ncost time: %f s" % cost_time