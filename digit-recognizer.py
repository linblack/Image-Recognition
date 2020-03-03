from keras import models
from keras import layers
from keras.datasets import mnist
import pandas as pd
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing

df_original = pd.read_csv(r'digit-recognizer\train.csv', sep=',')
df_submission = pd.read_csv(r'digit-recognizer\test.csv', sep=',')

df_image = df_original.iloc[:,1:]
df_label = df_original.iloc[:,0]

df_train_images = df_image.iloc[:36000]
df_train_images = df_train_images.values
df_train_images = df_train_images.reshape((36000, 784, 1))
train_images = df_train_images.astype('float32') / 255

df_train_labels = df_label.iloc[:36000]
train_labels = to_categorical(df_train_labels)

df_test_images = df_image.iloc[36000:]
df_test_images = df_test_images.values
df_test_images = df_test_images.reshape((6000, 784, 1))
test_images = df_test_images.astype('float32') / 255

df_test_labels = df_label.iloc[36000:]
test_labels = to_categorical(df_test_labels)

submission_images = df_submission.values
submission_images = submission_images.reshape((28000, 784, 1))
submission_images = submission_images.astype('float32') / 255

# model = models.Sequential()
# model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(784,1)))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Conv1D(64, 3, activation='relu'))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Conv1D(64, 3, activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
# model.save('digit_recognizer_1.h5')
model = models.load_model('digit_recognizer_1.h5')
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('test_acc:',test_acc)
prediction = model.predict_classes(submission_images)
print(prediction[:10])
pd.DataFrame({'ImageID':list(range(1, len(prediction)+1)),
              'Label':prediction}).to_csv('Digit_Recognizer_submission_3.csv', index=False, header=True)