#include "utils.h"

std::vector<std::vector<double>> Utils::getDataFromFile(std::string filePath) {
  std::vector<std::vector<double>> data;

  std::ifstream file(filePath);

  std::string line;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::vector<double> dataRow;
      std::string oneData;
      std::stringstream ss(line);

      while (std::getline(ss, oneData, ',')) {
        dataRow.push_back(std::stof(oneData));
      }
      data.push_back(dataRow);
    }
  } else {
    std::cerr << "Can not open a file " << filePath << std::endl;
  }
  return data;
}
// i want a vector of matrix
void Utils::saveWeightToFile(std::string pathToFile,
                             std::vector<std::shared_ptr<Matrix>> weights) {
  nlohmann::json json = {};
  std::vector<std::vector<std::vector<double>>> vectorWeightMatrixs;

  for (std::size_t i = 0; i < weights.size(); ++i) {
    vectorWeightMatrixs.push_back(weights.at(i)->getMatrix());
  }

  json["weights"] = vectorWeightMatrixs;
  std::ofstream writeToFile(pathToFile);
  if (writeToFile.is_open()) {
    writeToFile << std::setw(4) << json << std::endl;
    writeToFile.close();
  } else {
    std::cerr << "Unable to open a file" << std::endl;
  }
}

std::vector<std::shared_ptr<Matrix>>
Utils::loadWeights(std::string pathToFile) {

  std::ifstream file(pathToFile);
  std::vector<std::vector<std::vector<double>>> vectorWeightMatrixs;
  std::vector<std::shared_ptr<Matrix>> readMatrix;

  if (!file.is_open()) {
    std::cerr << "Error openning file" << std::endl;
    return {};
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string fileContent = buffer.str();

  nlohmann::json data = nlohmann::json::parse(fileContent);
  vectorWeightMatrixs = data["weights"];

  std::size_t matrixRow = 0;
  for (std::size_t index = 0; index < vectorWeightMatrixs.size(); ++index) {
    auto matrix = std::make_shared<Matrix>(
        vectorWeightMatrixs.at(index).size(),
        vectorWeightMatrixs.at(index).at(matrixRow).size(), false);
    for (std::size_t row = 0; row < vectorWeightMatrixs.at(index).size();
         ++row) {
      for (std::size_t column = 0;
           column < vectorWeightMatrixs.at(index).at(row).size(); ++column) {
        matrix->setValue(row, column,
                         vectorWeightMatrixs.at(index).at(row).at(column));
      }
    }
    readMatrix.push_back(matrix);
    matrixRow++;
  }
  file.close();
  return readMatrix;
}

void Utils::missingInputArgumentPredict() {
  std::cout << "Use: ./predict </path/to/the/predict.json>" << std::endl;
}

void Utils::missingInputArgumentTrain() {
  std::cout << "Use: ./train </path/to/the/config.json>" << std::endl;
}
