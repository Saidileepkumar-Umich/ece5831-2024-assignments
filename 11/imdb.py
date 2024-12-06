import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

class Imdb:
    def __init__(self, max_features=10000, maxlen=500):
        self.max_features = max_features
        self.maxlen = maxlen
        self.model = None

    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=self.max_features)
        self.x_train = pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = pad_sequences(self.x_test, maxlen=self.maxlen)

    def build_model(self):
        self.model = models.Sequential([
            layers.Embedding(self.max_features, 128, input_length=self.maxlen),
            layers.LSTM(32),
            layers.Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def train(self, epochs=3, batch_size=32):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)
