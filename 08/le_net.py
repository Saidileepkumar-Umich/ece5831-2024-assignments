
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2

class LeNet:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self._create_lenet()
        self._compile()

    def _create_lenet(self):
        # Building a simple CNN model similar to LeNet architecture
        self.model = models.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])

    def _compile(self):
        # Compile the model with optimizer, loss, and metrics
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _preprocess(self, images):
        # Preprocess input images to the model's required input shape and normalize
        processed_images = [cv2.resize(img, (28, 28)) for img in images]  # Resize to 28x28
        processed_images = np.array(processed_images).astype("float32") / 255.0  # Normalize
        if processed_images.ndim == 3:  # Add channel dimension if grayscale
            processed_images = np.expand_dims(processed_images, -1)
        return processed_images

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
        # Convert labels to categorical format for training
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        # Train the model
        history = self.model.fit(x_train, y_train_cat, validation_data=(x_val, y_val_cat), 
                                 epochs=epochs, batch_size=batch_size)
        return history

    def save(self, model_path_name):
        # Save the model in .keras format
        self.model.save(f"{model_path_name}.keras")
        print(f"Model saved as {model_path_name}.keras")

    def load(self, model_path_name):
        # Load a saved model
        self.model = tf.keras.models.load_model(f"{model_path_name}.keras")
        print(f"Model loaded from {model_path_name}.keras")

    def predict(self, images):
        # Predict the classes of the given images
        images = self._preprocess(images)  # Preprocess images before prediction
        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)
