from keras.datasets import imdb
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, LSTM, Bidirectional
import os
import numpy as np
import matplotlib.pyplot as plt

# imdb_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\aclImdb_v1.tar\aclImdb_v1\aclImdb'
# train_dir = os.path.join(imdb_dir, 'train')
# labels = []
# texts = []
#
# for label_type in ['neg','pos']:
#     dir_name = os.path.join(train_dir, label_type)
#     for fname in os.listdir(dir_name):
#         if fname[-4:] == '.txt':
#             f = open(os.path.join(dir_name, fname), encoding='utf8')
#             texts.append(f.read())
#             f.close()
#             if label_type == 'neg':
#                 labels.append(0)
#             else:
#                 labels.append(1)

max_words = 10000
max_len = 500
training_samples = 200
validation_samples = 10000
batch_size = 32

# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(texts)   #建立字典
# sequence = tokenizer.texts_to_sequences(texts)  #將文字轉成整數list
# word_index = tokenizer.word_index   #字典的詞彙表{'the':1,'and':2...}
# data = pad_sequences(sequence, maxlen=max_len)
# labels = np.asarray(labels)
# indices = np.arange(data.shape[0])  #指標[0-24999]
# np.random.shuffle(indices)  #指標洗牌[14482...14003]
# data = data[indices]
# labels = labels[indices]
# x_train = data[:training_samples]
# y_train = labels[:training_samples]
# x_val = data[training_samples:training_samples+validation_samples]
# y_val = labels[training_samples:training_samples+validation_samples]
#
# glove_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\aclImdb_v1.tar\aclImdb_v1\aclImdb\glove.6B'
# embeddings_index = {}
# f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32') #座標
#     embeddings_index[word] = coefs
# f.close()
#
# embedding_matrix = np.zeros((max_words, max_len))
# for word, i in word_index.items():
#     if i < max_words:
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector

(train_x, train_y),(test_x, test_y) = imdb.load_data(num_words=max_words)
train_x = preprocessing.sequence.pad_sequences(train_x, maxlen=max_len)
test_x = preprocessing.sequence.pad_sequences(test_x, maxlen=max_len)
#
model = Sequential()
model.add(Embedding(max_words, 32))
model.add(Bidirectional(LSTM(32)))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_x, train_y, epochs=10, batch_size=128, validation_split=0.2)
model.save_weights('imdb_glove_3.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training_acc')
plt.plot(epochs, val_acc, 'b', label='Validation_acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()