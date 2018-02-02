from keras import layers
from keras import models
from keras.datasets.boston_housing import load_data


(train_data, train_labels), (test_data, test_labels) = load_data()
train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)
test_data = (test_data - train_data.mean(axis=0)) / train_data.std(axis=0)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(train_data, train_labels, epochs=80, batch_size=1)

results = model.evaluate(test_data, test_labels)
print(results)
