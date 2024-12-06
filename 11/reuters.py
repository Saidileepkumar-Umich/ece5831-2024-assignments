from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

class Reuters:
    def __init__(self, num_classes=46, max_words=10000):
        self.num_classes = num_classes
        self.max_words = max_words
        self.model = None

    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = reuters.load_data(num_words=self.max_words)
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.x_train = self.tokenizer.sequences_to_matrix(self.x_train, mode='binary')
        self.x_test = self.tokenizer.sequences_to_matrix(self.x_test, mode='binary')
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def build_model(self):
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.max_words,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, epochs=5, batch_size=64):
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
