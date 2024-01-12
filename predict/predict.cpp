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
    Utils::missingInputArgumentPredict();
    exit(-1);
  }

  Predict predict;

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
      predict.numOfNeuronsActivationFunction.push_back(
          {item["numberOfNeurons"], item["activationFunction"]});
    }

    predict.loadWeightsPath = data["weightsFile"];
    predict.testDataPath = data["testData"];
    predict.testLabelDataPath = data["testLabelData"];

  } catch (nlohmann::json::parse_error &e) {
    std::cerr << "JSON parsing error: " << e.what() << std::endl;
  }

  std::unique_ptr<NeuralNetwork> NN = std::make_unique<NeuralNetwork>(predict);
  NN->predict();

  return 0;
}