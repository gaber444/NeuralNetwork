#include "layer.h"

Layer::Layer(int size, std::string activatedType) : m_size(size) {

  try {
    if (activatedType.empty()) {
      for (std::size_t i = 0; i < m_size; ++i) {
        std::shared_ptr<Neuron> n = std::make_shared<Neuron>(0.0);
        m_neurons.push_back(n);
      }
    } else {
      for (std::size_t i = 0; i < m_size; ++i) {
        std::shared_ptr<Neuron> n =
            std::make_shared<Neuron>(0.0, activatedType);
        m_neurons.push_back(n);
      }
    }
  } catch (std::exception &err) {
    std::cerr << "Error creating neurons: " << err.what() << std::endl;
  }
}

void Layer::setValueOfNeuron(int i, double value) {
  try {
    m_neurons.at(i)->setValue(value);
  } catch (const std::runtime_error &err) {
    std::cerr << "Error setting neuron value: " << err.what() << std::endl;
  }
}

std::shared_ptr<Matrix> Layer::layerAsMatrix() {
  auto inputLayerMatrix = std::make_shared<Matrix>(1, m_neurons.size(), false);
  for (std::size_t i = 0; i < m_neurons.size(); ++i) {
    inputLayerMatrix->setValue(0, i, m_neurons.at(i)->getValue());
  }
  return inputLayerMatrix;
}

std::shared_ptr<Matrix> Layer::layerDerivedAsMatrix() {
  auto inputLayerDerivedMatrix =
      std::make_shared<Matrix>(1, m_neurons.size(), false);
  for (std::size_t i = 0; i < m_neurons.size(); ++i) {
    inputLayerDerivedMatrix->setValue(0, i, m_neurons.at(i)->getDerivedValue());
  }
  return inputLayerDerivedMatrix;
}

std::shared_ptr<Matrix> Layer::layerActivatedAsMatrix() {
  auto inputLayerActivatedMatrix =
      std::make_shared<Matrix>(1, m_neurons.size(), false);
  for (std::size_t i = 0; i < m_neurons.size(); ++i) {
    inputLayerActivatedMatrix->setValue(0, i,
                                        m_neurons.at(i)->getActivatedValue());
  }
  return inputLayerActivatedMatrix;
}

std::vector<std::shared_ptr<Neuron>> Layer::getNeurons() { return m_neurons; }