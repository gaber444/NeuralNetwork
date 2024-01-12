#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H

#include <algorithm>
#include <map>
#include <vector>

#include "layer.h"
#include "matrix.h"
#include "utils.h"

struct Topology {
  int numberOfNeuronsInLayer;
  std::string activationFunction;
};

struct Params {
  std::vector<Topology> numOfNeuronsActivationFunction;
  double bias;
  double learningRate;
  double momentum;
  std::string trainingDataPath;
  std::string labelDataPath;
};

struct Predict {
  std::vector<Topology> numOfNeuronsActivationFunction;
  double bias;
  std::string loadWeightsPath;
  std::string testDataPath;
  std::string testLabelDataPath;
};

class NeuralNetwork {
public:
  /**
   * @brief Construct a new Neural Network object for training.
   *
   * @param Struct with initializing data for training.
   *
   */
  NeuralNetwork(Params &params);

  /**
   * @brief Construct a new Neural Network object for prediction.
   *
   * @param Struct with initializing data for predicting.
   */
  NeuralNetwork(Predict &predict);

  /**
   * @brief Destroy the Neural Network object.
   *
   */
  virtual ~NeuralNetwork() = default;

  /**
   * @brief Set the Values To Neurons at input Layer.
   *
   * @param std::vector<double> values at neurons in input layer.
   */
  void setValuesToNeuronsInputLayer(std::vector<double> valuesAtNeurons);

  /**
   * @brief Takes layer and then takes some specific neuron in this layer
   * and set new value to this neuron.
   *
   * @param indexLayer which layer in neural network.
   * @param indexNeuron which neuron in the layer(indexLayer).
   * @param value new at the neuron.
   */
  void setNeuronValue(int indexLayer, int indexNeuron, double value);

  /**
   * @brief Prints matrix of weights for each layer to console.
   *
   */
  void printMatrixForEachLayer();

  /**
   * @brief Get the Neuron values in form of matrix.
   *
   * @param index  which layer in neural network.
   * @return std::shared_ptr<Matrix> pointer to a metric of neuron values.
   */
  std::shared_ptr<Matrix> getNeuronMatrix(int index);

  /**
   * @brief Get the Activated Neuron values in form of matrix.
   *
   * @param index  which layer in neural network.
   * @return std::shared_ptr<Matrix> pointer to a metric of the Activated neuron
   * values.
   */
  std::shared_ptr<Matrix> getActivatedNeuronMatrix(int index);

  /**
   * @brief Get the Derived Neuron values in form of matrix.
   *
   * @param index which layer in neural network.
   * @return std::shared_ptr<Matrix> pointer to a metric of the Derived neuron.
   * values.
   */
  std::shared_ptr<Matrix> getDerivedNeuronMatrix(int index);

  /**
   * @brief Get the matrix of weights for the layer at index.
   *
   * @param index of a layer
   * @return std::shared_ptr<Matrix> pointer to a metric of weights.
   */
  std::shared_ptr<Matrix> getWeightMatrix(int index);

  /**
   * @brief It multiplies values on the neuron and matrix of weights
   * in this specific layer and than makes a vector from this new
   * calculated matrix of multiplication so that we can do next multiplication
   * until we get to the last layer.(neurons on the left * weights to the right
   * = neuron to the right).
   *
   */
  void feedForward();
  /**
   * @brief Get the Total Error of the neural network. This is the sum of the
   * errors in vector m_errors.
   *
   * @return double total error of the network.
   */
  double getTotalError() const;

  /**
   * @brief Get error for each neuron on the output layer.
   *
   * @return std::vector<double> of errors on the output layer.
   */
  std::vector<double> getErrors() const;

  /**
   * @brief Set the target of neural network. The value to which you want to
   * get close as possible.
   *
   * @param std::vector<double> vector of target values.
   */
  void setCurrentTarget(std::vector<double> target);

  /**
   * @brief Set errors function where we must already have predefined m_target.
   *
   */
  void setErrors();

  /**
   * @brief Executes the back propagation algorithm.
   *
   * This function performs the back propagation algorithm in a neural network,
   * adjusting the weights based on calculated errors.
   */
  void backPropagation();

  /**
   * @brief Trains the neural network for a specified number of epochs.
   *
   * This function initiates the training process for the neural network,
   * iterating over a given number of epochs to adjust weights and biases
   * based on input data and expected outputs.
   *
   * @param numberOfEpoch The number of training epochs to execute.
   */
  void train(int numberOfEpoch);

  /**
   * @brief It predicts which thing it should be on the given data.
   * The highest value on the neuron on the output layer gives
   * prediction.
   */
  void predict();

  /**
   * @brief get vector of weights matrices
   *
   * @return std::vector<std::shared_ptr<Matrix>> vector of weight for neural
   * network.
   */
  std::vector<std::shared_ptr<Matrix>> getWeightMatrices();

private:
  /** Number of neurons in each layer. */
  std::vector<int> m_topology;
  /** Number of layers in neural network.*/
  std::vector<std::shared_ptr<Layer>> m_layers;
  /** Number(topology - 1) of matrices with weights in neural network.*/
  std::vector<std::shared_ptr<Matrix>> m_weightMatrices;
  /** Number of layers in neural network.*/
  int m_topologySize;
  /** Values at neurons at input layer. */
  std::vector<double> m_inputLayer;
  /** Should have the same size as the output layer. Means that we will be
   * learning our neurons network on this targets, this will be the ones that
   * you know.
   */
  std::vector<double> m_target;
  /** Current error or total error for the current network.*/
  double m_error;
  /** Bias is an additional input to nodes/neurons, providing flexibility
   * to model complex relationships between inputs and outputs. It enables
   * the neural network to fit non-linear patterns in data.
   */
  double m_bias;
  /** Scaling factor for updating the parameters as matrix.
   * This will be a scalar.
   */
  std::shared_ptr<Matrix> m_learningRate;
  /** For accelerating gradient descent.
   * This is also a scalar.
   */
  std::shared_ptr<Matrix> m_momentum;
  /** Present error for each neuron in the output layer.*/
  std::vector<double> m_errors;
  /** Stores error at each interation.*/
  std::vector<double> m_historicalErrors;
  /** all gradients calculated in neural network*/
  std::vector<std::shared_ptr<Matrix>> m_gradientMatrices;
  /** Vector new  matrix weights, calculated as old matrix of weights - delta
   * weight matrix*/
  std::vector<std::shared_ptr<Matrix>> m_updatedWeightsMatrices;
  /** this are used for back propagation*/
  std::vector<double> m_derivedErrors;
  /** training data from a file */
  std::vector<std::vector<double>> m_trainingData;
  /** label data from a file*/
  std::vector<std::vector<double>> m_labelsData;
  /** data for prediction*/
  std::vector<std::vector<double>>
      m_predictionData; // try with float see how it goes
  /** labels to check prediction*/
  std::vector<std::vector<double>> m_labelsPredictionData;
};

#endif // _NEURAL_NETWORK_H