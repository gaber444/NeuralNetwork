#ifndef _NEURON_H
#define _NEURON_H

#include <cmath>
#include <memory>
#include <string>
#include <vector>

class Neuron {
public:
  /**
   * @brief Construct a new Neuron object
   *
   */
  Neuron(double value, std::string activatedType = "");

  /**
   * @brief Destroy the Neuron object.
   *
   */
  virtual ~Neuron() = default;

public:
  /**
   * @brief Fast Sigmoid function.
   * f(x) = x / (1 + |x|)
   *
   */
  void activate();

  /**
   * @brief Hyperbolic tan.
   *
   */
  void activateTANH();

  /**
   * @brief Rectified linear unit. if value on neuron is
   * bigger than zero, value dose not change, otherwise
   * value is set to zero.
   *
   */
  void activateRELU();

  /**
   * @brief Derived of Sigmoid function.
   * f'(x) = f(x) * ( 1 - f(x))
   *
   */
  void derive();

  /**
   * @brief Derived of hyperbolic tan.
   *
   */
  void deriveTANH();

  /**
   * @brief Derived of Rectified linear unit.
   *
   */
  void deriveRELU();

  // Getters

  /**
   * @brief Get the Value object.
   *
   * @return double , value of neuron
   */
  double getValue();

  /**
   * @brief Get the Activated Value object.
   *
   * @return double normalized value of neuron
   */
  double getActivatedValue();

  /**
   * @brief Get the Derived Value object.
   *
   * @return double derived value of neuron.
   */
  double getDerivedValue();

  /**
   * @brief Set the Value object. In case of Input layer,
   * when you want to set specific value on the neuron.
   *
   * @param value to be set on a neuron.
   */
  void setValue(double value);

private:
  /** Each neuron has exactly one value.*/
  double m_value;
  /** Normalized value of that neuron, on interval [0,1]*/
  double m_activatedValue;
  /** Derived value of neuron. */
  double m_derivedValue;
  /**activated type*/
  std::string m_activatedType;
};

#endif // _NEURON_H