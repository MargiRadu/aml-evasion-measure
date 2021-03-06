import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model


class AbstractModel:
    def __init__(self):
        self.num_epochs = 21
        self.batch_size = 32
        self.model = None
        self.optimizer = None

    def predict(self, data):
        return self.model.predict(data)

    def update(self, update_data, update_labels):
        self.model.fit(x=update_data,
                       y=update_labels,
                       batch_size=self.batch_size,
                       epochs=self.num_epochs,
                       shuffle=True,
                       verbose=0)

    def train(self, train_data, train_labels, num_epochs=None, batch_size=None, file_name=None):
        if batch_size is None:
            batch_size = self.batch_size
        if num_epochs is None:
            num_epochs = self.num_epochs

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        self.model.fit(x=train_data,
                       y=train_labels,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

        if file_name is not None:
            self.model.save(file_name)

    def distilation_train(self, trained_model, train_data, T=0.2, num_epochs=None, batch_size=None, file_name=None):
        """
        Trains the model using soft labels generated by the provided model.
        :param train_data:      Initial training data
        :param trained_model:   A model trained on the initial data.
        :param T:               Temperature coeff (0.0-1.0)
        """

        # Get the labels.
        train_labels = trained_model.predict(train_data)

        # Soften the labels.
        uniform_p = 0.1
        train_labels -= (train_labels - uniform_p) * T

        # Train using the new labels.
        self.train(train_data, train_labels, num_epochs, batch_size, file_name)

    def test(self, test_data, test_labels):
        metrics = self.model.evaluate(test_data, test_labels, verbose=0)
        return {metric_name: metrics[i] for i, metric_name in enumerate(self.model.metrics_names)}

    def compute_confusion_matrix(self, test_data, test_labels):
        confusion = np.zeros(shape=(10,10), dtype=np.int32)
        predictions = np.argmax(self.model.predict(test_data), axis=1)
        labels = np.argmax(test_labels, axis=1)

        for predicted, true in zip(predictions, labels):
            confusion[predicted, true] += 1

        return confusion


class MLPModel(AbstractModel):
    def __init__(self, restore):
        """
        Sets up the model used for crafting/evaluating poisonous examples.
        :param architecture:    {default, shallow, narrow} Specifies the architecture used.
        :param restore:         Path to serialized model.
        """
        super().__init__()

        # Setup training params.
        self.num_epochs = 15
        self.batch_size = 128
        self.lr = 0.0009
        self.decay = 1e-6
        self.momentum = 0.9

        # Setup model.
        model = Sequential()
        # Input
        model.add(Conv2D(1, (1,1), input_shape=(28, 28, 1)))
        model.add(Flatten())
        # Dense 1
        model.add(Dense(512, activation='relu'))
        # Dense 2
        model.add(Dense(1024, activation='relu'))
        # Output
        model.add(Dense(10))
        model.add(Activation('softmax'))

        self.model = model
        if restore:
            if os.path.isfile(restore):
                self.model = load_model(restore)
                self.optimizer = self.model.optimizer
            else:
                self.model = None
                print(f'Could not load model from {restore}')
        else:
            self.optimizer = Adam(lr=self.lr, decay=self.decay)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.optimizer,
                           metrics=['accuracy'])


class MNISTModel(AbstractModel):
    def __init__(self, architecture, restore):
        """
        Sets up the model used for crafting/evaluating poisonous examples.
        :param architecture:    {default, shallow, narrow} Specifies the architecture used.
        :param restore:         Path to serialized model.
        """
        super().__init__()

        # Setup training params.
        self.num_epochs = 15
        self.batch_size = 128
        self.lr = 0.0009
        self.decay = 1e-6
        self.momentum = 0.9

        # Setup model.
        if architecture == 'narrow':
            dense_nodes = 256
        else:
            dense_nodes = 512

        if architecture == 'sigmoid':
            activation = 'sigmoid'
        else:
            activation = 'relu'

        model = Sequential()

        # Input + Conv 1
        model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=(28, 28, 1)))

        # Conv 2 + MaxPool + DO
        model.add(Conv2D(32, kernel_size=(3, 3), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv 3 + MaxPool
        if architecture != 'shallow':
            model.add(Conv2D(64, kernel_size=(3, 3), activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten + Dense + DO
        model.add(Flatten())
        model.add(Dense(dense_nodes, activation=activation))

        # Output
        model.add(Dense(10))
        model.add(Activation('softmax'))

        self.model = model
        if restore:
            if os.path.isfile(restore):
                self.model = load_model(restore)
                self.optimizer = self.model.optimizer
            else:
                self.model = None
                print(f'Could not load model from {restore}')
        else:
            self.optimizer = Adam(lr=self.lr, decay=self.decay)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.optimizer,
                           metrics=['accuracy'])


class CIFARModel(AbstractModel):
    def __init__(self, architecture, restore):
        """
        Sets up the model used for crafting/evaluating poisonous examples.
        :param architecture:    {default, shallow, narrow} Specifies the architecture used.
        :param restore:         Path to serialized model.
        """
        super().__init__()

        # Setup training params.
        self.num_epochs = 21
        self.batch_size = 32
        self.lr = 0.0009
        self.decay = 1e-6
        self.momentum = 0.9

        # Setup model.
        if architecture == 'narrow':
            dense_nodes = 512
        else:
            dense_nodes = 1024

        model = Sequential()

        # Input + Conv 1
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

        # Conv 2 + MaxPool + DO
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Conv 3 + MaxPool
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv 4 + MaxPool + DO
        if architecture != 'shallow':
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

        # Flatten + Dense + DO
        model.add(Flatten())
        model.add(Dense(dense_nodes, activation='relu'))

        model.add(Dropout(0.5))

        # Output
        model.add(Dense(10, activation='softmax'))

        self.model = model
        if restore:
            if os.path.isfile(restore):
                self.model = load_model(restore)
                self.optimizer = self.model.optimizer
            else:
                self.model = None
                print(f'Could not load model from {restore}')
        else:
            self.optimizer = Adam(lr=self.lr, decay=self.decay)
        self.model.compile(loss= "categorical_crossentropy",
                           optimizer=self.optimizer,
                           metrics=['accuracy'])