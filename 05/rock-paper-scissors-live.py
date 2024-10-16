import cv2
import numpy as np
import tensorflow as tf

def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

model = tf.keras.models.load_model('keras_model.h5')

class_names = load_labels('labels.txt')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(frame, (224, 224)) 
    img = np.array(img, dtype=np.float32) / 255.0  
    img = np.expand_dims(img, axis=0) 

    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissors', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
