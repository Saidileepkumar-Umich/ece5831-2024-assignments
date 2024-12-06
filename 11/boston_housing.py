from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

class BostonHousing:
    def __init__(self):
        self.model = None

    def prepare_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        self.mean = self.x_train.mean(axis=0)
        self.std = self.x_train.std(axis=0)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

    def build_model(self):
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def train(self, epochs=20, batch_size=32):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)
