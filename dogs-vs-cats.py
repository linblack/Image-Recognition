import os, shutil
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50
import matplotlib.pyplot as plt

original_dataset_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\dogs-vs-cats\train'
base_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\dogs-vs-cats\small'

if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

train_dog_dir = os.path.join(train_dir, 'dogs')
if not os.path.isdir(train_dog_dir):
    os.mkdir(train_dog_dir)

train_cat_dir = os.path.join(train_dir, 'cats')
if not os.path.isdir(train_cat_dir):
    os.mkdir(train_cat_dir)

validation_dog_dir = os.path.join(validation_dir, 'dogs')
if not os.path.isdir(validation_dog_dir):
    os.mkdir(validation_dog_dir)

validation_cat_dir = os.path.join(validation_dir, 'cats')
if not os.path.isdir(validation_cat_dir):
    os.mkdir(validation_cat_dir)

test_dog_dir = os.path.join(test_dir, 'dogs')
if not os.path.isdir(test_dog_dir):
    os.mkdir(test_dog_dir)

test_cat_dir = os.path.join(test_dir, 'cats')
if not os.path.isdir(test_cat_dir):
    os.mkdir(test_cat_dir)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cat_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cat_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cat_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dog_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dog_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dog_dir, fname)
#     shutil.copyfile(src, dst)

# print('訓練用的貓圖片張數:',len(os.listdir(train_cat_dir)))
# print('訓練用的狗圖片張數:',len(os.listdir(train_dog_dir)))
# print('驗證用的貓圖片張數:',len(os.listdir(validation_cat_dir)))
# print('驗證用的狗圖片張數:',len(os.listdir(validation_dog_dir)))
# print('測試用的貓圖片張數:',len(os.listdir(test_cat_dir)))
# print('測試用的狗圖片張數:',len(os.listdir(test_dog_dir)))
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(150,150,3))

model = models.Sequential()
model.add(conv_base)
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())     #(樣本數,4,4,512) => (樣本數,8192)
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'res5c_branch2c':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=20, class_mode='binary')

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
model.save('cats_and_dogs_small_res2.h5')
# model = models.load_model('cats_and_dogs_small_res2.h5')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test_acc:',test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()