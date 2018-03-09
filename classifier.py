from keras import layers
from keras import models
from keras.utils import to_categorical


class Classifier:

    def __init__(self, data, labels, n_classes=2, n_labels=1,
                 train_size=0.7):
        self.data = data
        self.labels = labels
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.train_size = train_size

    def prepare_data(self):
        train_data = self.data[:int(self.train_size * len(self.data))]
        test_data = self.data[int(self.train_size * len(self.data)):]
        train_labels = self.labels[:int(self.train_size * len(self.labels))]
        test_labels = self.labels[int(self.train_size * len(self.labels)):]

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        return train_data, train_labels, test_data, test_labels

    def model(self):
        train_data, train_labels, test_data, test_labels = self.prepare_data()

        network = models.Sequential()
        network.add(layers.Dense(units=512,
                                 activation='relu',
                                 input_shape=(train_data[1],)))
        network.add(layers.Dense(train_labels.shape[1], activation='softmax'))
        network.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        network.fit(train_data, train_labels, epochs=5, batch_size=128)

        return network

    def evaluate_model(self):
        train_data, train_labels, test_data, test_labels = self.prepare_data()

        return self.model().evaluate(test_data, test_labels)
