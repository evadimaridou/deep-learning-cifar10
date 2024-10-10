# Neural Network Implementation for CIFAR-10 Classification

## Overview

This project implements and experiments with neural networks for image classification on the CIFAR-10 dataset. The implementation includes both a custom neural network from scratch using Numpy and models developed using PyTorch, including a fully connected (dense) network and a convolutional neural network (CNN). The CIFAR-10 dataset consists of 60,000 images across 10 classes, and the goal is to categorize each image into one of these classes.

## Contents

- **`neural_network_from_scratch.py`**: Implements a multi-layer perceptron (MLP) using only Numpy, without any pre-built deep learning libraries.
- **`pytorch_fc.py`**: A PyTorch implementation of a fully connected neural network to perform the same classification task.
- **`pytorch_conv.py`**: A PyTorch implementation of a convolutional neural network (CNN) to improve classification accuracy for image-based data.

## Project Structure

- **Data Loading and Preprocessing**: CIFAR-10 data is loaded, split, and preprocessed for training, validation, and testing. Data normalization and one-hot encoding are applied.
- **Numpy Implementation**: A neural network from scratch is implemented, including data flow, backpropagation, and gradient descent. Relu and Softmax are used as activation functions.
- **PyTorch Implementations**: Two versions of neural networks are implemented in PyTorch:
  - A dense neural network using fully connected layers.
  - A convolutional neural network (CNN) with convolutional and max-pooling layers to improve performance for image classification.

## Key Features

### 1. Neural Network from Scratch

- **Layers**: Implemented with custom classes for each layer.
- **Backpropagation and Gradient Descent**: Manually implemented to optimize network weights.
- **Performance**: Tested with different learning rates and architectures. Achieved up to 49.37% testing accuracy.

### 2. PyTorch Fully Connected Network

- **Training and Validation Split**: Data is split into training and validation sets for model evaluation.
- **Optimization**: Utilized Stochastic Gradient Descent (SGD) with different learning rates.
- **Performance**: Achieved similar accuracy as the custom implementation but with faster training times.

### 3. PyTorch Convolutional Neural Network (CNN)

- **Convolutional Layers**: Added to capture spatial features of the images.
- **Pooling**: Max pooling layers used to reduce the spatial dimensions and extract key features.
- **Testing Accuracy**: Achieved up to 82.18% accuracy, outperforming the dense models.

## Experiments

Multiple experiments were performed to evaluate the effect of different architectures, learning rates, optimizers, and the inclusion of convolutional layers:

1. **Basic MLP (from scratch)**: Using three hidden layers, achieving \~47% testing accuracy.
2. **Dense PyTorch Model**: Comparable performance to the scratch model, with faster training.
3. **CNN (PyTorch)**: Improved testing accuracy to 82% with convolutional layers.
4. **Effect of Learning Rate**: Lower learning rates improved model stability but increased training time.
5. **Optimizer Experiments**: Tested Adam and SGD, with different impacts on overfitting and training times.

## Usage

- **Numpy Implementation**: Run the script to train the neural network from scratch.
  ```sh
  python neural_network_from_scratch.py
  ```
- **PyTorch Dense Network**: Train the fully connected model using PyTorch.
  ```sh
  python pytorch_fc.py
  ```
- **PyTorch Convolutional Network**: Train the CNN model.
  ```sh
  python pytorch_conv.py
  ```

## Results Summary

| Experiment | Model                   | Testing Accuracy | Training Time    |
| ---------- | ----------------------- | ---------------- | ---------------- |
| 1          | From Scratch (MLP)      | 47.38%           | 1 hour 28 mins   |
| 2          | PyTorch Dense (SGD)     | 45.43%           | 46 mins          |
| 3          | From Scratch (Lower LR) | 49.37%           | 1 hour 30 mins   |
| 6          | CNN (PyTorch)           | 56.2%            | 1 hour 47 mins   |
| 8          | Advanced CNN (PyTorch)  | 82.18%           | 12 hours 46 mins |

## Conclusion

The use of convolutional layers significantly improved classification accuracy, demonstrating their effectiveness in image-related tasks. Compared to simpler dense layers, the CNN models achieved much better accuracy, highlighting the advantage of capturing spatial features inherent in image data.

## Acknowledgments

- The CIFAR-10 dataset was used for training and validation.
- References include PyTorch tutorials and various online deep learning resources.

