from keras import layers
from keras import models
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np

from utils import vectorise_sequences


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)

x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512,
          validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print(results)
