# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from configs import save_model_path
import utils_cut_NN

offline = False # False

(train_x, train_y, test_x, test_y, date, routes) = utils_cut_NN.load_data(offline=offline)
print(len(train_x), len(train_y), len(test_x), len(test_y))

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(60, input_dim=30, activation='relu'))
model.add(Dense(30, input_dim=60, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(6))

# 训练，当offline = False时忽略validation
if offline:
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0002))
    MAPE_best=0.15800
    for e in range(1000):
        print('Times=%d'%e)
        model.fit(train_x, train_y, nb_epoch=100, verbose=1, batch_size=2000, validation_data=(test_x, test_y), shuffle=True)
        predict_y = model.predict(test_x)
        MAPE_new=utils_cut_NN.get_MAPE(date, routes, predict_y, test_y)
        if(MAPE_new<MAPE_best):
            model.save_weights(save_model_path + "best_model.h5")
            MAPE_best = MAPE_new
    print('MAPE_best = %0.5f'%MAPE_best)

else:
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.000001))
    model.load_weights(save_model_path + "best_model.h5")
    predict_y = model.predict(test_x)
    utils_cut_NN.write_prediction(date, routes, predict_y)
