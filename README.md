# Digit_Recognition_Neural_Network_FromScratch
This project implements a simple fully connected feedforward neural network from scratch to classify digits from the MNIST dataset. The neural network is built using basic Python and numpy, without any deep learning frameworks such as TensorFlow or PyTorch.

Key Features:
The network consists of three hidden layers:

Input Layer: Accepts the pixel values of the 28x28 grayscale images (784 pixels).
Two Hidden Layers: Each consisting of 20 neurons and ReLU activation function.
Output Layer: Predicts the digit classes (0-9) using the softmax activation function.
Forward Propagation: Implemented with ReLU activation for the hidden layers and softmax for the output.

Backward Propagation: Gradients are computed using the chain rule with respect to weights and biases, and parameters are updated using gradient descent.

Optimization: Gradient descent with adjustable learning rate (alpha) and iterations.

# Digit Recognizer Neural Network

This repository contains a Python implementation of a simple neural network built from scratch using `numpy` to recognize handwritten digits from the MNIST dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Network Architecture](#network-architecture)
- [Usage](#usage)
- [Key Functions](#key-functions)
- [Contributing](#contributing)

## Overview
This project aims to classify images of handwritten digits (0-9) using a feedforward neural network. The neural network is built from scratch using Python and `numpy`, and no external deep learning libraries are used.

The dataset used is from the Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition, which contains images of handwritten digits in 28x28 pixel format.

## Dataset
The model is trained on the MNIST dataset, where:
- **Training Data:** 42,000 images of digits, each represented by 784 pixel values (28x28 image) and their corresponding label.
- **Test Data:** 1,000 images from the dataset used for testing model performance.

## Network Architecture
The neural network consists of the following:
- **Input Layer:** Accepts 784 pixel values (28x28 image).
- **Two Hidden Layers:** Each consisting of 20 neurons with ReLU activation.
- **Output Layer:** 10 neurons corresponding to the digits (0-9) using the softmax function for classification.

The architecture can be visualized as:

Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output Layer (Softmax)


## Usage

### Prerequisites
- Python 3.x
- `numpy` and `pandas` libraries
- A CSV file containing the MNIST dataset.

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/digit-recognizer.git
   cd digit-recognizer
2. Install the required dependencies (if not already installed):
    ```bash
      pip install numpy pandas matplotlib
    
3. Download the dataset from Kaggle and place it in the same directory or specify the path in the code.

4. Run the script:
   python digit_recognizer.py

## Contributing
Feel free to fork this project and submit pull requests. Suggestions for improvements are always welcome!

---

### Key Points to Mention:
1. **Training Process:** The model goes through the process of forward propagation, backpropagation, and gradient descent to update the weights and biases over multiple iterations.
2. **Accuracy Calculation:** The model calculates accuracy every 10 iterations to track performance during training.
3. **Further Improvements:** You can suggest future work or improvements, such as experimenting with different neural network architectures, hyperparameter tuning, or integrating with deep learning libraries like TensorFlow.

Let me know if you'd like to add more specific details or features to the `README`!
