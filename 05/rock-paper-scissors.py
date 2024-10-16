import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2

def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

model = tf.keras.models.load_model('keras_model.h5')

class_names = load_labels('labels.txt')

def classify_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_resized = cv2.resize(img, (224, 224)) 
    img_normalized = np.array(img_resized, dtype=np.float32) / 255.0 
    img_expanded = np.expand_dims(img_normalized, axis=0)  

    predictions = model.predict(img_expanded)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    confidence_score = predictions[0][class_idx]

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    print(f"Class: {prediction_label}")
    print(f"Confidence Score: {confidence_score:.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python rock-paper-scissors.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_image(image_path)
