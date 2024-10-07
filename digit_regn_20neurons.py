import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and preprocess data
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # Shuffle the data to randomize training and testing examples

# Split into test and training data
data_test = data[0:1000].T  # Transpose for easier manipulation, test data with 1000 samples
Y_test = data_test[0]  # First row contains labels (correct digits)
X_test = data_test[1:n]  # Remaining rows contain pixel values
X_test = X_test / 255  # Normalize pixel values to be between 0 and 1

data_train = data[1000:m].T  # Training data (remaining 41,000 samples)
Y_train = data_train[0]  # First row contains labels for training
X_train = data_train[1:n]  # Pixel values for training data
X_train = X_train / 255  # Normalize pixel values
_, m_train = X_train.shape

# Initialize weights and biases for the neural network
#20 neurons in hidden layer will be corresponded here
def init_params():
    w1 = np.random.randn(20, 784) * np.sqrt(2 / 784)
    w2 = np.random.randn(20, 20) * np.sqrt(2 / 20)
    w3 = np.random.randn(10, 20) * np.sqrt(2 / 20)
    b1 = np.zeros((20, 1))
    b2 = np.zeros((20, 1))
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3

# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z -= np.max(Z, axis=0)  # Numerical stability
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

# Derivative of ReLU for backpropagation
def der_ReLU(Z):
    return Z > 0

# One-hot encoding of labels
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Forward propagation
def forward_propagation(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

# Backward propagation
def backward_propagation(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y):
    one_hot_Y = one_hot(Y)
    dz3 = a3 - one_hot_Y
    dw3 = 1 / m * dz3.dot(a2.T)
    db3 = 1 / m * np.sum(dz3, axis=1, keepdims=True)
    dz2 = w3.T.dot(dz3) * der_ReLU(z2)
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * der_ReLU(z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2, dw3, db3

# Update weights and biases
def update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3
    return w1, b1, w2, b2, w3, b3

# Get predictions
def get_predictions(a3):
    return np.argmax(a3, axis=0)

# Calculate accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Training the model using gradient descent
def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, X)
        dw1, db1, dw2, db2, dw3, db3 = backward_propagation(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y)
        w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        if i % 10 == 0:
            predictions = get_predictions(a3)
            print(f'Iteration {i}, Accuracy: {get_accuracy(predictions, Y)}')
    return w1, b1, w2, b2, w3, b3

# Test accuracy on test data
def test_accuracy(X_test, Y_test, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, X_test)
    predictions = get_predictions(a3)
    accuracy = get_accuracy(predictions, Y_test)
    print(f'Test Accuracy: {accuracy}')
    return accuracy

# Train the model and test on validation data
w1, b1, w2, b2, w3, b3 = gradient_descent(X_train, Y_train, 0.25, 500)
test_accuracy(X_test, Y_test, w1, b1, w2, b2, w3, b3)
#around 96 perecent accuracy is achieved on test data
