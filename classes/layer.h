#ifndef _LAYER_H
#define _LAYER_H

#include "matrix.h"
#include "neuron.h"

class Layer {
public:
  /**
   * @brief Construct a new Layer object
   *
   * @param size, number of neurons in a layer.
   * @param activatedType, name of the function by which
   * derived and activated value will be calculated on the
   * neurons in this layer. Empty is default and is for
   * Sigmoid function.
   */
  Layer(int size, std::string activatedType = "");

  /**
   * @brief Destroy the Layer object
   *
   */
  virtual ~Layer() = default;

  /**
   * @brief Set the Value Of Neuron in a layer.
   *
   * @param i position of neuron in a layer.
   * @param value of neuron
   */
  void setValueOfNeuron(int i, double value);

  /**
   * @brief Convert layer into (1 x values).
   * For easier multiplication.
   *
   * @return std::unique_ptr<Matrix> Pointer to converted matrix.
   */
  std::shared_ptr<Matrix> layerAsMatrix();

  /**
   * @brief Convert layer into (1 x derived values)
   *
   * @return std::shared_ptr<Matrix> Pointer to converted matrix.
   */
  std::shared_ptr<Matrix> layerDerivedAsMatrix();

  /**
   * @brief Convert layer into (1 x activated values)
   *
   * @return std::shared_ptr<Matrix> Pointer to converted matrix.
   */
  std::shared_ptr<Matrix> layerActivatedAsMatrix();

  /**
   * @brief Get all the neurons in a layer.
   *
   * @return std::vector<std::shared_ptr<Neuron>> Vector of the neuron in a
   * Layer.
   */
  std::vector<std::shared_ptr<Neuron>> getNeurons();

private:
  /** Number of neurons in a layer.*/
  int m_size;
  //** Vector of all neurons in a layer. */
  std::vector<std::shared_ptr<Neuron>> m_neurons;
};

#endif // _LAYER_H