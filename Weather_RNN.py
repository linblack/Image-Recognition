import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay -1
    i = min_index + lookback
    while i:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)  #(1440,200000,128)產生128個整數[1440-200000]
        else:
            #200001 > 200000
            if i + batch_size >= max_index:
                i = min_index + lookback    #i=1440
            rows = np.arange(i, min(i+batch_size, max_index))   #1440,1568
            i = i + len(rows)   #i=1568
        samples = np.zeros((len(rows),lookback//step, data.shape[-1]))    #(128,240,14)
        targets = np.zeros((len(rows),))    #(128,)
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)    #(0,1440,6)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

data_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\jena_climate_2009_2016'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

mean = float_data[:200000].mean(axis=0)
float_data = float_data - mean
std = float_data[:200000].std(axis=0)
float_data = float_data / std

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step, batch_size=batch_size)
val_steps = (300000-200001-lookback) // batch_size
test_steps = (len(float_data)-300001-lookback) // batch_size

model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback//step, float_data.shape[-1])))   #(240,14)
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()