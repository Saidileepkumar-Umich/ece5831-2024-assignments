import cv2
import numpy as np
from tensorflow.keras.models import load_model

class Mnist:
    def __init__(self, model_path):
        self.model = load_model("./MNIST_keras_CNN.h5") 

    def predict(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
        # Predict digit
        prediction = self.model.predict(image)
        predicted_digit = np.argmax(prediction)
        return predicted_digit
