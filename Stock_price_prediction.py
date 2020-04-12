import twstock
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers

def storefile(data, filename, one_value='false'):
    if one_value == 'true':
        data = list(map(lambda x:[x],data))
    with open(filename, 'w', newline='') as f:
        mywrite = csv.writer(f)
        if one_value == 'true':
            for i in data:
                mywrite.writerow(i)
        else:
            mywrite.writerow(data)

def get_stock_data(name, year, month):
    tsmc = twstock.Stock(name)
    # print(tsmc.price)   #31天收盤價[遠~近]
    tsmc_his = pd.DataFrame(tsmc.fetch_from(year,month))  #2008/5
    tsmc_his = tsmc_his.drop(['capacity'], axis=1)
    tsmc_his = tsmc_his.drop(['turnover'], axis=1)
    # print(tsmc_his)

    #----------capacity--------------#
    tsmc_his_capacity_5 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 5)) #31-5+1
    tsmc_his_capacity_10 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 10))   #31-10+1
    tsmc_his_capacity_20 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 20))
    tsmc_his_capacity_30 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 30))
    tsmc_his_capacity_40 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 40))
    tsmc_his_capacity_60 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 60))
    tsmc_his_capacity_120 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 120))
    tsmc_his_capacity_240 = pd.DataFrame(tsmc.moving_average(tsmc.capacity, 240))
    #----------capacity--------------#

    #----------ma_bias_ratio--------------#
    tsmc_his_ma_bias_ratio_5_10 = pd.DataFrame(tsmc.ma_bias_ratio(5, 10))
    tsmc_his_ma_bias_ratio_10_20 = pd.DataFrame(tsmc.ma_bias_ratio(10, 20))
    tsmc_his_ma_bias_ratio_20_30 = pd.DataFrame(tsmc.ma_bias_ratio(20, 30))
    tsmc_his_ma_bias_ratio_30_40 = pd.DataFrame(tsmc.ma_bias_ratio(30, 40))
    tsmc_his_ma_bias_ratio_40_60 = pd.DataFrame(tsmc.ma_bias_ratio(40, 60))
    tsmc_his_ma_bias_ratio_60_120 = pd.DataFrame(tsmc.ma_bias_ratio(60, 120))
    tsmc_his_ma_bias_ratio_120_240 = pd.DataFrame(tsmc.ma_bias_ratio(120, 240))
    #----------ma_bias_ratio--------------#

    tsmc_his_all_v3 = pd.concat([tsmc_his, tsmc_his_capacity_5, tsmc_his_capacity_10, tsmc_his_capacity_20, tsmc_his_capacity_30, tsmc_his_capacity_40, tsmc_his_capacity_60, tsmc_his_capacity_120,
                                 tsmc_his_capacity_240, tsmc_his_ma_bias_ratio_5_10, tsmc_his_ma_bias_ratio_10_20, tsmc_his_ma_bias_ratio_20_30, tsmc_his_ma_bias_ratio_30_40, tsmc_his_ma_bias_ratio_40_60,
                                 tsmc_his_ma_bias_ratio_60_120, tsmc_his_ma_bias_ratio_120_240], axis=1)
    tsmc_his_all_v3.to_csv('tsmc_his_all_test.csv', header=0, index=0)

def evaluate_native_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:,-1,3]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=8, step=2):
    if max_index is None:
        max_index = len(data) - delay - 1   #delay為label

    i = min_index + lookback
    #while 1 永久循環
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)  #[750 480 317 476 563 1092 654 371]
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index)) #[240 241 242 243 244 245 246 247]
            i = i+len(rows) #240+8
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))   #(8, 60, 21)
        targets = np.zeros((len(rows), ))   #[0. 0. 0. 0. 0. 0. 0. 0.]
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)  #range(953, 1193, 2) => 953, 955...
            # indices_list = []
            # for a in indices:
            #     indices_list.append(a)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][3]   #data[1193+10][3] => close
        yield samples, targets  #return and iterable
        # return samples, targets

# get_stock_data('2330', 2019, 4)
# storefile(tsmc_his_capacity, 'tsmc_his_capacity.csv', 'true')
# pd.set_option('display.max_columns', None)  #顯示所有列
# pd.set_option('display.max_rows', None) #顯示所有行

tsmc_his_all = pd.read_csv('tsmc_his_all_v4.csv', header=0)
date = tsmc_his_all['date']
tsmc_his_all = tsmc_his_all.drop(['date'], axis=1).to_numpy().astype('float')
mean = tsmc_his_all[:1800].mean(axis=0) #取1600筆為train data
tsmc_his_all = tsmc_his_all - mean
std = tsmc_his_all[:1800].std(axis=0)
tsmc_his_all = tsmc_his_all / std

feature_list = ['open','high','low','close','change','transaction','capacity_5','capacity_10','capacity_20','capacity_30','capacity_40','capacity_60','capacity_120','capacity_240','ma_bias_ratio_5_10',
               'ma_bias_ratio_10_20','ma_bias_ratio_20_30','ma_bias_ratio_30_40','ma_bias_ratio_40_60','ma_bias_ratio_60_120','ma_bias_ratio_120_240']
tsmc_test = pd.read_csv('tsmc_his_all_test.csv', names=['date','open','high','low','close','change','transaction','capacity_5','capacity_10','capacity_20','capacity_30','capacity_40','capacity_60','capacity_120',
                                                        'capacity_240','ma_bias_ratio_5_10','ma_bias_ratio_10_20','ma_bias_ratio_20_30','ma_bias_ratio_30_40','ma_bias_ratio_40_60','ma_bias_ratio_60_120','ma_bias_ratio_120_240'])
tsmc_test_all = []
for feature in feature_list:
    tsmc_test_last = tsmc_test[tsmc_test[feature].notnull()][feature].iloc[[-2]]  #建立list->for loop->concat->predict[21]
    tsmc_test_all.append(tsmc_test_last.values[0])
tsmc_test_all_na = np.asarray(tsmc_test_all).reshape((1,21))
tsmc_test_all_na = (tsmc_test_all_na - mean) / std
tsmc_test_all_na = np.asarray(tsmc_test_all_na).reshape((1,1,21))

# lookback = 120  #240->120
# step = 2    #5->2
# delay = 10  #20->10
# batch_size = 8
#
# train_gen = generator(tsmc_his_all, lookback=lookback, delay=delay, min_index=0, max_index=1800, shuffle=True, step=step, batch_size=batch_size)    #1200->1600->1800
# val_gen = generator(tsmc_his_all, lookback=lookback, delay=delay, min_index=1801, max_index=2000, step=step, batch_size=batch_size)
# test_gen = generator(tsmc_his_all, lookback=lookback, delay=delay, min_index=2001, max_index=None, step=step, batch_size=batch_size)
# val_steps = (2000-1801-lookback) // batch_size
# test_steps = (len(tsmc_his_all)-2001-lookback) // batch_size

# evaluate_native_method()
# model = models.Sequential()
# # model.add(layers.Flatten(input_shape=(lookback // step, tsmc_his_all.shape[-1])))
# # model.add(layers.Dense(32, activation='relu'))
# # model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, tsmc_his_all.shape[-1])))
# # model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
# model.add(layers.Bidirectional(layers.GRU(128), input_shape=(None, tsmc_his_all.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=optimizers.RMSprop(), loss='mae')
# history = model.fit_generator(train_gen, steps_per_epoch=225, epochs=3, validation_data=val_gen, validation_steps=val_steps)   #careful fit & fit_generator
# model.save('SPP_3.h5')

model = models.load_model('SPP_3.h5')
prediction = model.predict(tsmc_test_all_na)
prediction_1 = (prediction * std[3]) + mean[3]
print(prediction)
print(prediction_1)

# acc = history.history['loss']
# val_acc = history.history['val_loss']
# epochs = range(1, len(acc)+1)
#
# plt.plot(epochs, acc, 'bo', label='Training_acc')
# plt.plot(epochs, val_acc, 'b', label='Validation_acc')
# plt.title('Training and Validation acc')
# plt.legend()
# plt.show()
