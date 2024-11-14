
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from le_net import LeNet

def preprocess_image(image_path):
    # Load, convert, resize, and preprocess image for LeNet model
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_data = np.array(image, dtype=np.float32)
    image_data = 255.0 - image_data  # Invert colors
    image_data = image_data / 255.0  # Normalize
    image_data = np.expand_dims(image_data, axis=(0, -1))  # Reshape for model input
    return image_data

def main(image_path, true_label):
    # Load the trained LeNet model
    lenet = LeNet()
    lenet.load("Mukkamala_cnn_model")
    
    # Preprocess and predict
    x = preprocess_image(image_path)
    predicted_label = lenet.predict([x])[0]
    
    # Show the image
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis("off")
    plt.show()
    
    # Print result
    if predicted_label == int(true_label):
        print(f"Success: Image {image_path} is recognized correctly as {true_label}.")
    else:
        print(f"Fail: Image {image_path} is for digit {true_label} but the inference result is {predicted_label}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test handwritten digit image with trained CNN model")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("true_label", type=int, help="True digit label of the image")
    args = parser.parse_args()
    main(args.image_path, args.true_label)
