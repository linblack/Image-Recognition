from keras import models
from keras import layers
from keras.datasets import mnist
import pandas as pd
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
from keras.applications import VGG16
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv(r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\digit-recognizer\train.csv', sep=',')
test = pd.read_csv(r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\digit-recognizer\test.csv', sep=',')

Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)

(x_train1, y_train1),(x_test1, y_test1) = mnist.load_data()
train1 = np.concatenate([x_train1, x_test1], axis=0)
Y_train1 = np.concatenate([y_train1, y_test1], axis=0)
X_train1 = train1.reshape(-1, 28*28)

X_train = X_train/255.0
test = test/255.0
x_train1 = X_train1/255.0
X_train = np.concatenate([X_train.values, X_train1])
Y_train = np.concatenate([Y_train.values, Y_train1])

X_train = X_train.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes=10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

model = models.Sequential()
model.add(layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (5,5), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))
#
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=(X_val, Y_val))
#
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()
#
model.save('digit_recognizer_4.h5')
# model = models.load_model('digit_recognizer_3.h5')
# # test_loss, test_acc = model.evaluate(test_images, test_labels)
# # print('test_acc:',test_acc)
prediction = model.predict_classes(test)
# # print(prediction[:10])
pd.DataFrame({'ImageID':list(range(1, len(prediction)+1)),
              'Label':prediction}).to_csv('Digit_Recognizer_submission_6.csv', index=False, header=True)