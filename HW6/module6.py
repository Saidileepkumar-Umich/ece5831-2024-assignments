
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

# Function to preprocess image and prepare it for model input
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))               # Resize to 28x28 pixels
    image_data = np.asarray(image, dtype=np.float32)
    image_data = 255.0 - image_data              # Invert colors
    image_data /= 255.0                          # Normalize
    return image_data.flatten().reshape(1, -1)   # Flatten and reshape for the model

def main(image_path, true_label):
    # Load the trained model
    with open("Mukkamala_mnist_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Preprocess the input image
    x = preprocess_image(image_path)
    
    # Predict using the model
    y = model.predict(x)
    predicted_label = np.argmax(y)
    
    # Show the image
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis("off")
    plt.show()
    
    # Print the result
    if predicted_label == int(true_label):
        print(f"Success: Image {image_path} is recognized correctly as {true_label}.")
    else:
        print(f"Fail: Image {image_path} is for digit {true_label} but the inference result is {predicted_label}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test handwritten digit image with trained model")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("true_label", type=int, help="True digit label of the image")
    args = parser.parse_args()
    main(args.image_path, args.true_label)
