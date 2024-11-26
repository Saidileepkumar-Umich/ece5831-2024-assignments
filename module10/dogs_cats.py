import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory

class DogsCats:
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')
    EPOCHS = 20

    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name, start_index, end_index):
        destination = self.BASE_DIR / subset_name
        destination.mkdir(parents=True, exist_ok=True)

        for class_name in self.CLASS_NAMES:
            (destination / class_name).mkdir(parents=True, exist_ok=True)

            for i in range(start_index, end_index):
                file_name = f"{class_name}.{i}.jpg"
                src_file = self.SRC_DIR / file_name
                dst_file = destination / class_name / file_name
                tf.io.gfile.copy(str(src_file), str(dst_file), overwrite=True)

    def _make_dataset(self, subset_name):
        dataset = image_dataset_from_directory(
            self.BASE_DIR / subset_name,
            label_mode='binary',
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE
        )
        return dataset

    def make_dataset(self):
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('valid')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        model = models.Sequential()

        # Data augmentation layers
        if augmentation:
            model.add(layers.RandomFlip('horizontal'))
            model.add(layers.RandomRotation(0.1))
            model.add(layers.RandomZoom(0.1))

        # Add convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.IMAGE_SHAPE))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer=optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Explicitly build the model
        model.build(input_shape=(None, *self.IMAGE_SHAPE))

        # Assign the model to the class attribute
        self.model = model
        

    def train(self, model_name):
        checkpoint = ModelCheckpoint(f"{model_name}", save_best_only=True, monitor='val_accuracy', mode='max')
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.EPOCHS,
            callbacks=[checkpoint]
        )

        self.plot_training_history(history)

    def plot_training_history(self, history):
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        prediction = self.model.predict(img_array)[0][0]
        label = 'dog' if prediction > 0.5 else 'cat'

        plt.imshow(img)
        plt.title(f"Prediction: {label}")
        plt.axis('off')
        plt.show()

# Usage Example in a Jupyter Notebook
# ds = DogsCats()
# ds.make_dataset_folders('valid', 0, 2400)
# ds.make_dataset_folders('train', 2400, 12000)
# ds.make_dataset_folders('test', 12000, 12500)
# ds.make_dataset()
# ds.build_network()
# ds.model.summary()
# ds.train("model.dogs-vs-cats.keras")
