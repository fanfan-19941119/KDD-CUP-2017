# coding=utf-8

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import utils

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#记录程序运行时间
import time
start_time = time.time()

#读入数据
(train_x, train_y, test_x, test_y, actual_time, date, routes) = utils.load_data(False)
print(len(train_x), len(train_y), len(test_x))
print(train_x.shape)

trainx, testx, trainy, scalerX, scalerY = utils.mscale(train_x, train_y, test_x)

# train_x,val_x, train_y, val_y = train_test_split(train_X, train_Y, test_size = 0.2, random_state=1)

# xgb_val = xgb.DMatrix(val_x,label=val_y)
# xgb_train = xgb.DMatrix(train_x, label=train_y)
# xgb_test = xgb.DMatrix(test)
# watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]


def modelfit(alg, train_X, train_Y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_X, label=train_Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    print cvresult.iloc[cvresult.shape[0] - 1]
    # cvresult.get_value()
    # Fit the algorithm on the data
    alg.fit(train_X, train_Y)

    # Predict training set:
    # pred_y = alg.predict(train_X)
    # dtrain_predprob = alg.predict_proba(pred_y)[:, 1]

    # Print model report:
    # print "\nModel Report"
    # print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(train_Y, dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

#Choose all predictors except target & IDcols
# predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=1000)

# modelfit(xgb1, train_X, train_Y)

#
# def myscore(y, py):
#     return np.mean(np.abs(py - y) / y)


param_test1 = {
 'max_depth':range(6,8,1),
 'min_child_weight':range(4,6,1)
}
gsearch1 = GridSearchCV(
    estimator = XGBRegressor(
        learning_rate =0.1,
        n_estimators=2,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:logistic',
        # nthread=4,
        # scale_pos_weight=1,
        seed=1000),
    param_grid = param_test1,
    n_jobs=4,
    iid=False,
    cv=5)
gsearch1.fit(trainx, trainy)

for _ in gsearch1.grid_scores_:
    print _
print gsearch1.best_params_, gsearch1.best_score_