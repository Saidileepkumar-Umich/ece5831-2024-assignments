
import numpy as np

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        if x.ndim == 2:
            x = x - np.max(x, axis=1, keepdims=True)
            x = np.exp(x)
            x /= np.sum(x, axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x
