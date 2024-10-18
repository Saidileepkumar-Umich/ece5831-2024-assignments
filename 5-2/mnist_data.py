import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class MnistData:
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.dataset_path = './mnist.pkl'
        self._init_dataset()

    def _download(self, url, filename):
        """ Placeholder function for downloading dataset files """
        print(f"Placeholder: Would download {filename} from {url}")

    def _download_all(self):
        """ Placeholder function for downloading all dataset files """
        print("Placeholder: Would download all MNIST files")
        # Keras takes care of downloading, so we don't need to actually implement this.

    def _load_images(self, filename):
        """ Placeholder for loading images from a local file (if not using Keras) """
        print(f"Placeholder: Would load images from {filename}")
        # Since we use Keras' built-in loader, this function is not needed.
        return np.zeros((60000, 784))  # Dummy return for illustration

    def _load_labels(self, filename):
        """ Placeholder for loading labels from a local file (if not using Keras) """
        print(f"Placeholder: Would load labels from {filename}")
        # Since we use Keras' built-in loader, this function is not needed.
        return np.zeros(60000)  # Dummy return for illustration

    def _create_dataset(self):
        """ Placeholder for creating the dataset from the downloaded files """
        print("Placeholder: Would create dataset by loading images and labels")
        self.train_images = self._load_images('train-images-idx3-ubyte.gz')
        self.train_labels = self._load_labels('train-labels-idx1-ubyte.gz')
        self.test_images = self._load_images('t10k-images-idx3-ubyte.gz')
        self.test_labels = self._load_labels('t10k-labels-idx1-ubyte.gz')
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(((self.train_images, self.train_labels), (self.test_images, self.test_labels)), f)

    def _init_dataset(self):
        """ Initialize dataset by loading from Keras or loading local files """
        if not os.path.exists(self.dataset_path):
            print("Dataset not found locally, loading from Keras...")
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        else:
            print("Dataset found, loading dataset from file...")
            with open(self.dataset_path, 'rb') as f:
                (self.train_images, self.train_labels), (self.test_images, self.test_labels) = pickle.load(f)

    def load(self):
        """ Load the dataset in a flattened format for consistency """
        train_images_flat = self.train_images.reshape(-1, 28*28)
        test_images_flat = self.test_images.reshape(-1, 28*28)
        return (train_images_flat, self.train_labels), (test_images_flat, self.test_labels)

    def _change_one_hot_label(self, labels, num_classes=10):
        """ Convert labels to one-hot encoding """
        one_hot_labels = np.zeros((labels.size, num_classes))
        one_hot_labels[np.arange(labels.size), labels] = 1
        return one_hot_labels

    def show_image(self, dataset_type='train', index=0):
        """ Display an image and its label """
        if dataset_type == 'train':
            image = self.train_images[index]
            label = self.train_labels[index]
        else:
            image = self.test_images[index]
            label = self.test_labels[index]

        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()
