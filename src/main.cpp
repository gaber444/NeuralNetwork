#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.h"
#include "neuralNetwork.h"
#include "neuron.h"
#include "nlohmann/json.hpp"

int main(int argc, char **argv) {

  if (argc != 2) {
    Utils::missingInputArgumentTrain();
    exit(-1);
  }

  Params params;
  int epoch = 0;
  std::string pathToSaveWeights;

  try {

    std::ifstream configFile(argv[1]);

    if (!configFile.is_open()) {
      std::cerr << "Error openning file." << std::endl;
      return 1;
    }
    std::stringstream buffer;
    buffer << configFile.rdbuf();
    std::string fileContent = buffer.str();

    nlohmann::json data = nlohmann::json::parse(fileContent);

    for (const auto &item : data["topology"]) {
      params.numOfNeuronsActivationFunction.push_back(
          {item["numberOfNeurons"], item["activationFunction"]});
    }

    params.bias = data["bias"];
    params.learningRate = data["learningRate"];
    params.momentum = data["momentum"];
    params.trainingDataPath = data["trainingData"];
    params.labelDataPath = data["labelData"];
    epoch = data["epoch"];
    pathToSaveWeights = data["weightsFile"];

  } catch (nlohmann::json::parse_error &e) {
    std::cerr << "JSON parsing error: " << e.what() << std::endl;
  }

  std::unique_ptr<NeuralNetwork> NN = std::make_unique<NeuralNetwork>(params);
  NN->train(epoch);
  // save weights
  Utils::saveWeightToFile(pathToSaveWeights, NN->getWeightMatrices());

  for (auto const &j : params.numOfNeuronsActivationFunction) {
    std::cout << "Activation function: " << j.activationFunction << std::endl;
    std::cout << "number of Neurons: " << j.numberOfNeuronsInLayer << std::endl;
  }

  std::cout << "bias: " << params.bias << std::endl;
  std::cout << "learning rate: " << params.learningRate << std::endl;
  std::cout << "momentum: " << params.momentum << std::endl;

  return 0;
}