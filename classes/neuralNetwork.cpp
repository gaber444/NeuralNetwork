#include "neuralNetwork.h"

NeuralNetwork::NeuralNetwork(Params &params)
    : m_error(0.0), m_bias(params.bias),
      m_momentum(std::make_shared<Matrix>(1, 1, false)),
      m_learningRate(std::make_shared<Matrix>(1, 1, false)) {

  m_topologySize = params.numOfNeuronsActivationFunction.size();
  m_momentum->setValue(0, 0, params.momentum);
  m_learningRate->setValue(0, 0, params.learningRate);

  for (auto const &numOfLayer : params.numOfNeuronsActivationFunction) {
    m_layers.push_back(std::make_shared<Layer>(
        numOfLayer.numberOfNeuronsInLayer, numOfLayer.activationFunction));
    m_topology.push_back(numOfLayer.numberOfNeuronsInLayer);
  }

  for (std::size_t numberOfMatrices = 0;
       numberOfMatrices < (m_topologySize - 1); numberOfMatrices++) {
    m_weightMatrices.push_back(
        std::make_shared<Matrix>(m_topology.at(numberOfMatrices),
                                 m_topology.at(numberOfMatrices + 1), true));
  }

  m_trainingData = Utils::getDataFromFile(params.trainingDataPath);
  m_labelsData = Utils::getDataFromFile(params.labelDataPath);
}

// Constructor for predicting.
NeuralNetwork::NeuralNetwork(Predict &predict)
    : m_error(0.0), m_bias(predict.bias),
      m_momentum(std::make_shared<Matrix>(1, 1, false)),
      m_learningRate(std::make_shared<Matrix>(1, 1, false)) {

  m_topologySize = predict.numOfNeuronsActivationFunction.size();

  for (auto const &numOfLayer : predict.numOfNeuronsActivationFunction) {
    m_layers.push_back(std::make_shared<Layer>(
        numOfLayer.numberOfNeuronsInLayer, numOfLayer.activationFunction));
    m_topology.push_back(numOfLayer.numberOfNeuronsInLayer);
  }

  m_weightMatrices = Utils::loadWeights(predict.loadWeightsPath);
  m_labelsPredictionData = Utils::getDataFromFile(predict.testLabelDataPath);
  m_predictionData = Utils::getDataFromFile(predict.testDataPath);
  std::cout << "in constructor,"
            << "predict size: " << m_predictionData.size() << std::endl;
}

void NeuralNetwork::setValuesToNeuronsInputLayer(
    std::vector<double> valuesAtNeurons) {
  m_inputLayer = valuesAtNeurons;
  for (std::size_t i = 0; i < valuesAtNeurons.size(); i++) {
    m_layers.at(0)->setValueOfNeuron(i, valuesAtNeurons.at(i));
  }
}

void NeuralNetwork::setNeuronValue(int indexLayer, int indexNeuron,
                                   double value) {
  m_layers.at(indexLayer)->setValueOfNeuron(indexNeuron, value);
}

void NeuralNetwork::printMatrixForEachLayer() {
  for (std::size_t i = 0; i < m_layers.size(); ++i) {
    std::cout << "Layer: " << i << std::endl;
    if (i == 0) {
      m_layers.at(i)->layerAsMatrix()->printMatrixValues();
    } else {
      m_layers.at(i)->layerActivatedAsMatrix()->printMatrixValues();
    }
    std::cout << "--------------------------------" << std::endl;
    if (i < (m_layers.size() - 1)) {
      std::cout << "Weight Matrix for layer: " << i << std::endl;
      getWeightMatrix(i)->printMatrixValues();
    }
    std::cout << "--------------------------------" << std::endl;
  }
}

std::shared_ptr<Matrix> NeuralNetwork::getNeuronMatrix(int index) {
  return m_layers.at(index)->layerAsMatrix();
}

std::shared_ptr<Matrix> NeuralNetwork::getActivatedNeuronMatrix(int index) {
  return m_layers.at(index)->layerActivatedAsMatrix();
}

std::shared_ptr<Matrix> NeuralNetwork::getDerivedNeuronMatrix(int index) {
  return m_layers.at(index)->layerDerivedAsMatrix();
}

std::shared_ptr<Matrix> NeuralNetwork::getWeightMatrix(int index) {
  return m_weightMatrices.at(index);
}

std::vector<std::shared_ptr<Matrix>> NeuralNetwork::getWeightMatrices() {
  return m_weightMatrices;
}

void NeuralNetwork::feedForward() {
  std::shared_ptr<Matrix> newMatrix = nullptr;
  for (std::size_t i = 0; i < m_layers.size() - 1; ++i) {
    if (i != 0) {
      newMatrix = getActivatedNeuronMatrix(i) * getWeightMatrix(i);
    } else {
      newMatrix = getNeuronMatrix(i) * getWeightMatrix(i);
    }
    for (std::size_t c_index = 0; c_index < newMatrix->getNumberOfColumns();
         c_index++) {
      setNeuronValue(i + 1, c_index, newMatrix->getValue(0, c_index) + m_bias);
    }
  }
}

double NeuralNetwork::getTotalError() const { return m_error; }

std::vector<double> NeuralNetwork::getErrors() const { return m_errors; }

void NeuralNetwork::setCurrentTarget(std::vector<double> target) {
  m_target = target;
}

void NeuralNetwork::setErrors() {
  if (m_target.size() == 0) {
    throw std::runtime_error("No defined target for this  NEURAL NETWORK.");
  }
  if (m_target.size() !=
      (m_layers.at(m_layers.size() - 1))->getNeurons().size()) {
    throw std::runtime_error(
        "Target size is not the same as the output LAYER SIZE.");
  }
  // Calculate error on the every neuron at the OUTPUT LAYER.
  m_error = 0.0;
  m_errors.clear();
  m_derivedErrors.clear();
  std::size_t indexOfOutputLayer = m_layers.size() - 1;
  // here we calculate error in one way, but there is more ways in which you
  // can calculate errors.
  for (std::size_t i = 0; i < m_target.size(); ++i) {
    double tempError = m_layers.at(indexOfOutputLayer)
                           ->getNeurons()
                           .at(i)
                           ->getActivatedValue() -
                       m_target.at(i);

    double err = 0.5 * pow(tempError, 2);
    m_errors.push_back(err);

    m_derivedErrors.push_back(2.0 * (m_layers.at(indexOfOutputLayer)
                                         ->getNeurons()
                                         .at(i)
                                         ->getActivatedValue() -
                                     m_target.at(i)));
    m_error += err;
  }
  // Store all global errors at each iteration of the neural network.
  m_historicalErrors.push_back(m_error / (indexOfOutputLayer + 1));
}

void NeuralNetwork::backPropagation() {
  // *****output to hidden layer *****
  int indexOutPutLayer = m_layers.size() - 1;
  std::shared_ptr<Matrix> derivedValuesOnOutputLayer =
      m_layers.at(indexOutPutLayer)->layerDerivedAsMatrix();

  std::shared_ptr<Matrix> gradient = std::make_shared<Matrix>(
      1, m_layers.at(indexOutPutLayer)->getNeurons().size(), false);

  for (std::size_t gg = 0; gg < m_derivedErrors.size(); ++gg) {
    gradient->setValue(0, gg,
                       derivedValuesOnOutputLayer->getValue(0, gg) *
                           m_derivedErrors.at(gg));
  }
  m_gradientMatrices.push_back(gradient);
  int lastHiddenLayer = indexOutPutLayer - 1;

  auto deltaWeightedMatrix =
      (m_layers.at(lastHiddenLayer)->layerActivatedAsMatrix())->transpose() *
      gradient;

  auto updatedWeights = (m_weightMatrices.at(lastHiddenLayer) * m_momentum) -
                        (deltaWeightedMatrix * m_learningRate);

  m_updatedWeightsMatrices.insert(m_updatedWeightsMatrices.begin(),
                                  updatedWeights);

  // ***** hidden to the input layer. ******
  for (std::size_t i = lastHiddenLayer; i > 0; --i) {

    auto derivedGradients =
        std::make_shared<Matrix>(1, m_layers.at(i)->getNeurons().size(), false);

    auto weightMatrix = m_weightMatrices.at(i);
    auto originalWeightMetrix = m_weightMatrices.at(i - 1);
    auto activatedHidden = m_layers.at(i)->layerActivatedAsMatrix();

    for (std::size_t r = 0; r < weightMatrix->getNumberOfRows(); ++r) {
      double sum = 0;
      for (std::size_t c = 0; c < weightMatrix->getNumberOfColumns(); ++c) {
        double p = m_gradientMatrices.back()->getValue(0, c) *
                   weightMatrix->getValue(r, c);
        sum += p;
      }
      double g = sum * activatedHidden->getValue(0, r);
      derivedGradients->setValue(0, r, g);
    }

    m_gradientMatrices.push_back(derivedGradients);

    auto leftMatrixOfNeurons =
        (i - 1) == 0 ? m_layers.at(0)->layerAsMatrix()
                     : m_layers.at(i - 1)->layerActivatedAsMatrix();

    auto deltaWeights = leftMatrixOfNeurons->transpose() * derivedGradients;

    auto newWeights =
        (originalWeightMetrix * m_momentum) - (deltaWeights * m_learningRate);

    m_updatedWeightsMatrices.insert(m_updatedWeightsMatrices.begin(),
                                    newWeights);
  }
  m_weightMatrices = m_updatedWeightsMatrices;
  m_updatedWeightsMatrices.clear();
}

void NeuralNetwork::train(int numberOfEpoch) {
  std::cout << "Start with training..." << std::endl;
  for (std::size_t i = 0; i < numberOfEpoch; ++i) {
    for (std::size_t index = 0; index < m_trainingData.size(); ++index) {

      setValuesToNeuronsInputLayer(m_trainingData.at(index));
      setCurrentTarget(m_labelsData.at(index));
      feedForward();
      setErrors();
      backPropagation();
    }
    std::cout << "Epoch " << i + 1 << ", total error: " << getTotalError()
              << std::endl;
  }
}

void NeuralNetwork::predict() {
  int correct = 0;
  for (std::size_t index = 0; index < m_predictionData.size(); ++index) {
    setValuesToNeuronsInputLayer(m_predictionData.at(index));
    setCurrentTarget(m_labelsPredictionData.at(index));

    feedForward();
    setErrors();

    int ss = m_layers.size() - 1;

    auto minElement = std::min_element(m_errors.begin(), m_errors.end());
    auto maxElement = std::max_element(m_labelsPredictionData.at(index).begin(),
                                       m_labelsPredictionData.at(index).end());
    std::size_t positionMIN = std::distance(m_errors.begin(), minElement);
    std::size_t positionMAX =
        std::distance(m_labelsPredictionData.at(index).begin(), maxElement);

    if (positionMIN == positionMAX) {
      correct++;
    } else {
      std::cout << "Data position: " << index << std::endl;
      std::cout << "predicted number: " << positionMIN << std::endl;
      std::cout << "actual number: " << positionMAX << std::endl;
    }
  }

  std::cout << "ACCURACY: " << (correct / m_predictionData.size()) * 100
            << std::endl;
}
