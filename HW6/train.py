
import numpy as np
import pickle
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
from sklearn.datasets import fetch_openml

# Set hyperparameters
iterations = 10000
batch_size = 16
learning_rate = 0.01

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data / 255.0  # Normalize pixel values
labels = mnist.target.astype(np.int)  # Convert labels to integers

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Set up the model and initialize
input_size = 784  # 28x28 images flattened
hidden_size = 50  # Hidden layer size
output_size = 10  # Number of classes (digits 0-9)

model = TwoLayerNetWithBackProp(input_size, hidden_size, output_size)

# Training loop
train_size = data.shape[0]
for i in range(iterations):
    # Mini-batch sampling
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = data[batch_mask]
    t_batch = one_hot_encode(labels[batch_mask])
    
    # Forward and backward passes
    loss = model.forward(x_batch, t_batch)
    model.backward()
    
    # Update parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        model.params[key] -= learning_rate * model.grads[key]
    
    # Print loss and accuracy at each epoch
    if i % (train_size // batch_size) == 0:
        print(f"Iteration {i}/{iterations}, Loss: {loss:.4f}")

# Save the model to a .pkl file
with open("Mukkamala_mnist_model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Model saved to trained_mnist_model.pkl")
