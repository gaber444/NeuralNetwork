#include "neuron.h"

Neuron::Neuron(double value, std::string activatedType)
    : m_value(value), m_activatedType(activatedType) {

  if (m_activatedType.empty()) {
    activate();
    derive();
  } else if (m_activatedType == "relu" || m_activatedType == "RELU") {
    activateRELU();
    deriveRELU();
  } else if (m_activatedType == "tanh" || m_activatedType == "TANH") {
    activateTANH();
    deriveTANH();
  } else {
    throw std::runtime_error("Invalid string for activation type\n");
  }
}

void Neuron::activate() {
  m_activatedValue = m_value / (1 + std::abs(m_value));
}

void Neuron::activateTANH() { m_activatedValue = tanh(m_value); }

void Neuron::activateRELU() {
  (m_value > 0) ? m_activatedValue = m_value : m_activatedValue = 0.0;
}

void Neuron::derive() {
  m_derivedValue = m_activatedValue * (1 - m_activatedValue);
}

void Neuron::deriveTANH() {
  m_derivedValue = (1.0 - (m_activatedValue * m_activatedValue));
}

void Neuron::deriveRELU() {
  (m_activatedValue > 0) ? m_derivedValue = 1.0 : m_derivedValue = 0.0;
}

double Neuron::getValue() { return m_value; }

double Neuron::getActivatedValue() { return m_activatedValue; }

double Neuron::getDerivedValue() { return m_derivedValue; }

void Neuron::setValue(double value) {
  m_value = value;

  if (m_activatedType.empty()) {
    activate();
    derive();
  } else if (m_activatedType == "relu" || m_activatedType == "RELU") {
    activateRELU();
    deriveRELU();
  } else if (m_activatedType == "tanh" || m_activatedType == "TANH") {
    activateTANH();
    deriveTANH();
  } else {
    throw std::runtime_error("Invalid string for activation type\n");
  }
}
