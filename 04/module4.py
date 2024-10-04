import numpy as np
from multilayer_perceptron import MultiLayerPerceptron

def main():
    # Dummy dataset (XOR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MultiLayerPerceptron(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

    print("Training...")
    mlp.train(X, y, epochs=10000)

    print("Testing...")
    for i in range(len(X)):
        output = mlp.forward(X[i])
        print(f"Input: {X[i]}, Predicted Output: {output}, True Output: {y[i]}")

if __name__ == "__main__":
    main()
