import pandas as pd
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.applications import ResNet50, VGG16
import os
import random

#字串均分，奇數加到前一個
s = input('input:')
l = len(s)
m = l//2
if l%2 > 0:
    m = m+1
print(s[:m],s[m:])

#返回前二跟後二字母，小於二顯示空字串
# s = input('input:')
# if len(s) < 2:
#     print('')
# else:
#     if len(s) > 3:
#         s = s[:2]+s[-2:]
#     print(s)

#與第一個字母相同，替換成*
# s=input('input:')
# for i in range(1,len(s)):
#    print(i)
#    if s[0]==s[i]:
#        s=s[:i]+'*'+s[i+1:]
# print(s)

#tuple & list
# a = (1,2,3,4)
# b = [1,2,3,4]
# b.append(5)

#隨機數
# a = random.randint(1,100)   #1~100間整數
# b = random.random()         #0~1間浮點數
# c = random.uniform(1,100)   #1~100間浮點數

#用set去重覆值
# list = [1,1,1,2,3,4]
# print(set(list))

#畫出☆三角形
# for i in range(1,4):
#     for j in range(i):
#         print('☆', end=' ')
#     print('\n')

# (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# print(train_images[0])
# train_images = train_images.reshape(60000,28,28,1)
# print(train_images.shape)
# print(train_images[0])
# train_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\venv\CMUH\trainset'
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(28,28), batch_size=32, class_mode='categorical')
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# history = model.fit_generator(train_generator, steps_per_epoch=1875, epochs=5)
# prediction = model.predict_classes(test_images)
# pd.DataFrame({'ImageID':list(range(1, len(prediction)+1)),
#               'Label':prediction}).to_csv('mnist_test.csv', index=False, header=True)
# (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
# for i in range(len(train_images)):
#     filename = 'C:/Users/Blake/PycharmProjects/Image_Recognition/venv/CMHU/trainset/' + str(train_labels[i]) + '/' +  str(i) + '.png'
#     cv2.imwrite(filename, train_images[i])

# df_train_images = pd.DataFrame(train_images.reshape((60000,784,1))[:,:,0])
# df_train_labels = pd.DataFrame(train_labels, columns=['target'])
# df_train = pd.concat([df_train_images,df_train_labels], axis=1)
# df_train[df_train['target'] == 0].to_csv(r'D:\Other1\Private\InterView\2020\CMUH\trainset\0\')

# iris = pd.read_csv(r'D:\Other1\Private\InterView\2020\CMUH\iris.csv', sep=',')
# X = iris.iloc[:,3:5]
# Y = iris.iloc[:,-1]
# x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=87)
# clf = SVC()
# clf.fit(x_train, y_train)
# prediction = clf.predict(x_test)
# pd.DataFrame({'ImageID':list(range(1, len(prediction)+1)),
#               'Label':prediction}).to_csv('iris_test.csv', index=False, header=True)
# iris = load_iris()
# df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
# df2 = pd.DataFrame(iris.target, columns=['target'])
# df3 = pd.concat([df1,df2], axis=1)
# df4 = df3[df3['target'] != 2]
# df4.to_csv(r'D:\Other1\Private\InterView\2020\CMUH\iris.csv')

# import cv2
# import numpy as np
# Vertebral_img = cv2.imread(r'D:\Other1\Private\InterView\2020\CMUH\Vertebral.png')
# gray_img = cv2.cvtColor(Vertebral_img, cv2.COLOR_BGR2GRAY)
# edge_image = cv2.Canny(gray_img, 50, 250)
# lines = cv2.HoughLinesP(edge_image, 3, np.pi/180, 60, minLineLength=40, maxLineGap=50)
# lines = lines[:,0,:]
# for x1, y1, x2, y2 in lines:
#     cv2.line(Vertebral_img, (x1,y1), (x2,y2), (0,0,255), 1)
# cv2.imshow('Vertebral', Vertebral_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# df = pd.read_csv(r'D:\Other1\Private\InterView\2020\CMUH\1081216.csv', sep=',')
# print(df[df['國道別'] == '國道一號']['小型車牌價'].mean(0))