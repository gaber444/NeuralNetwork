# Neural Network Project Description

## Overview
This project presents a versatile and customizable neural network framework designed for both experimentation and application in various fields of machine learning and artificial intelligence. The core feature of this framework is its configurability, allowing users to specify the architecture of the neural network through a user-friendly configuration file.

## Key Features

### Dynamic Neural Network Architecture
Customizable Layers and Neurons: Users can define the number of layers and the number of neurons in each layer. This flexibility allows for the creation of both simple and complex neural network architectures tailored to specific tasks.

**Config File Setup:** The entire neural network structure, including the number of layers and neurons, is defined in an easily editable JSON configuration file. This design choice makes it simple to modify the network architecture without altering the code.

### Activation Functions

**Multiple Activation Function Options:** The framework supports various activation functions for the neurons. 
Users can choose from:
    **ReLU (Rectified Linear Unit):**
    Effective for non-linear transformations with the advantage of not activating all neurons at the same time.
    **Tanh (Hyperbolic Tangent):**
    Useful for handling negative inputs, providing a scaled output.
    **Sigmoid:**
    The default function, offering a smooth gradient and working well for probabilities.
**Per-Layer Activation Function:** 
Each layer's activation function can be individually specified in the configuration file, offering high customization for different network behaviors.

## How to Use

### Neural Network Configuration Guide
This guide provides an overview of the usage of the configuration files for setting up and running the neural network in our application.

#### Configuration Files Overview

**Testing Configuration:**
Used to specify the network topology and parameters for testing the neural network.
**Training Configuration:**
Used to define the network topology and parameters for training the neural network.

#### Testing Configuration File
This file contains the settings used during the testing phase of the neural network.

**Sample Training Configuration for MNIST data:**

Below is an example of the JSON configuration file used to set up the neural network's architecture and training parameters:

```json
{
    "topology": [
        {
            "numberOfNeurons": 784,
            "activationFunction": "relu"
        },
        {
            "numberOfNeurons": 428,
            "activationFunction": "relu"
        },
        {
            "numberOfNeurons": 128,
            "activationFunction": "tanh"
        },
        {
            "numberOfNeurons": 10,
            "activationFunction": ""
        }
    ],
    "bias": 1.0,
    "learningRate": 0.05,
    "momentum": 1.0,
    "epoch": 3,
    "trainingData": "/path/to/train100.csv",
    "labelData": "/path/to/train100_label.csv",
    "weightsFile": "/path/to/weightsMNIST.json"
}
```
#### Explanation of Parameters

**topology:** Defines the structure of the neural network. Each entry in the array represents a layer in the network.
**numberOfNeurons:** The number of neurons in the layer.
**activationFunction:** The activation function used in the layer. Options are "relu", "tanh", or "" for the default sigmoid function.
**bias:** The bias value applied to neurons.
**learningRate:** The rate at which the network learns during training.
**momentum:** The momentum factor applied to the learning process.
**epoch:** The number of complete passes through the training dataset.
**trainingData:** The path to the CSV file containing the training data.
**labelData:** The path to the CSV file containing the labels for the training data.
**weightsFile:** The path to the JSON file where the network's learned weights will be stored after training.

Below is an example of the JSON configuration file for setting up the neural network's testing parameters:

```json
{
    "topology": [
        {
            "numberOfNeurons": 784,
            "activationFunction": "relu"
        },
        {
            "numberOfNeurons": 428,
            "activationFunction": "relu"
        },
        {
            "numberOfNeurons": 128,
            "activationFunction": "tanh"
        },
        {
            "numberOfNeurons": 10,
            "activationFunction": ""
        }
    ],
    "bias": 1.0,
    "weightsFile": "/path/to/weightsMNIST.json",
    "testData": "/path/to/test10.csv",
    "testLabelData": "/path/to/test10_label.csv"
}
```
**topology:** Same as in training json file.
**numberOfNeurons:** Same as in training json file.
**activationFunction:** Same as in training json file.
**bias:** Same as in training json file.
**weightsFile:** Path to the JSON file containing the pre-trained weights of the network.
**testData:** Path to the CSV file containing the test data.
**testLabelData:** Path to the CSV file containing the test data labels.

#### Usage

**To use these configurations:**

Place the configuration file in the config directory.

**For training run:**
```bash
        ./train /path/to/configFile/config/train.json
```

**For testing/predicting run:**
```bash
        ./predict /path/to/configFile/config/predict.json
```

#### Data
The data folder in our project contains the MNIST dataset, a widely used resource in the field of machine learning for handwritten digit recognition. This dataset is pre-processed and normalized, distributed across several .csv files for easy use in training and testing the neural network. Here's a breakdown of the contents:

##### Training Data
**train100.csv and train100_label.csv:** These files contain a small subset of 100 samples from the MNIST dataset. train100.csv holds the feature data (handwritten digit images), while train100_label.csv contains the corresponding labels (the actual digits each image represents).
**train.csv and train_label.csv:** These are the comprehensive training datasets, containing 60,000 samples. train.csv provides the feature data, and train_label.csv includes the corresponding labels. This larger dataset is ideal for deep training of the neural network.

##### Testing Data
**test.csv and test_label.csv:** Comprising 10,000 samples, these files are used exclusively for testing the neural network. test.csv contains the test images, and test_label.csv includes the correct labels. This dataset is crucial for evaluating the network's performance and accuracy in digit recognition.
