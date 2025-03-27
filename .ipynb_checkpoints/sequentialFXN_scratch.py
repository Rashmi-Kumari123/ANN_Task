

#### Implementation of sequential() method of sklearn  ################################
import numpy as np

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Derivative of Binary Cross-Entropy Loss
def binary_cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Weight Initialization
np.random.seed(42)
input_size = x_train.shape[1]  # type: ignore
hidden_size = 32
output_size = 1

# Initialize Weights and Biases
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, hidden_size))

W3 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
b3 = np.zeros((1, hidden_size))

W4 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b4 = np.zeros((1, output_size))

# Training with Stochastic Gradient Descent
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Forward Propagation
    Z1 = np.dot(x_train, W1) + b1   # type: ignore
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)

    Z4 = np.dot(A3, W4) + b4
    A4 = sigmoid(Z4)  # Output

    # Compute Loss
    loss = binary_cross_entropy(y_train, A4)  # type: ignore

    # Backpropagation
    dA4 = binary_cross_entropy_derivative(y_train, A4)   # type: ignore
    dZ4 = dA4 * (A4 * (1 - A4))  # Sigmoid derivative
    dW4 = np.dot(A3.T, dZ4)
    db4 = np.sum(dZ4, axis=0, keepdims=True)

    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * relu_derivative(Z3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(x_train.T, dZ1)        #type: ignore
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Gradient Descent Updates
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")




########################## XOR problem     #######################################

import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # Input Data (features)


y = np.array([[0], [1], [1], [0]])  # Expected Output (labels)

# Weights and Bias Initialization
np.random.seed(4)
W1 = np.random.randn(2, 4)  # Input to Hidden Layer weights
b1 = np.zeros((1, 4))       # Hidden Layer Bias

W2 = np.random.randn(4, 1)  # Hidden to Output Layer weights
b2 = np.zeros((1, 1))       # Output Layer Bias (Gradient descent automatically adjusts biases)

# Activation Function (ReLU and Sigmoid)
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward Pass (Manually Computing Output)
hidden_layer = relu(np.dot(X, W1) + b1)
output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

print("Output Predictions:", output_layer)
