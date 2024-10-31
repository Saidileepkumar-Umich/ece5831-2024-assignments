import sys
import cv2
import matplotlib.pyplot as plt
from mnist import Mnist

def main(image_path, actual_digit):
    actual_digit = int(actual_digit)
    mnist_model = Mnist('./MNIST_keras_CNN.h5')
    predicted_digit = mnist_model.predict(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    plt.show()

    if predicted_digit == actual_digit:
        print(f"Success: Image {image_path} is for digit {actual_digit} and is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_path} is for digit {actual_digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    args = sys.argv[1:3]
    if len(args) != 2:
        print("Usage: python module5-3.py <image_path> <actual_digit>")
        sys.exit(1)
    main(*args)
