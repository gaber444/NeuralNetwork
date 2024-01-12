#ifndef _UTILS_H
#define _UTILS_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.h"

class Utils {
public:
  /**
   * @brief Get the data from a file.
   *
   * @param filePath path to a data file.
   * @return std::vector<std::vector<double>> all data that is gathered from a
   * file.
   */
  static std::vector<std::vector<double>> getDataFromFile(std::string filePath);

  /**
   * @brief After training it saves weight to the .json file.
   *
   * @param pathToFile in which weights will be saved.
   */
  static void saveWeightToFile(std::string pathToFile,
                               std::vector<std::shared_ptr<Matrix>> weights);
  /**
   * @brief Load weights from a file.
   *
   * @param pathToFile file with weights
   * @return std::vector<std::shared_ptr<Matrix>> vector of weight Matrces.
   */
  static std::vector<std::shared_ptr<Matrix>>
  loadWeights(std::string pathToFile);

  /**
   * @brief Print correct use of predicting.
   *
   */
  static void missingInputArgumentPredict();

  /**
   * @brief Print correct use of training.
   *
   */
  static void missingInputArgumentTrain();
};

#endif // _UTILS_H