#ifndef _MATRIX_H
#define _MATRIX_H

#include <exception>
#include <iostream>
#include <random>
#include <vector>

#include "nlohmann/json.hpp"

class Matrix {
public:
  /**
   * @brief Construct a new Matrix object.
   *
   * @param numberOfRows
   * @param numberOfColumns
   * @param isRandom If 'true', the matrix is populated with random numbers,
   * otherwise, it retains the same default values.
   */
  Matrix(int numberOfRows, int numberOfColumns, bool isRandom);

  /**
   * @brief Destroy the Matrix object.
   *
   */
  virtual ~Matrix() = default;

  /**
   * @brief Return transpose of the matrix.
   *
   * @return Matrix* pointer to a object Matrix.
   */
  std::shared_ptr<Matrix> transpose();

  /**
   * @brief Set the Value at specific position((row, column)) in matrix.
   *
   * @param row
   * @param column
   * @param value
   */
  void setValue(int row, int column, double value);

  /**
   * @brief Get the the specific value at position (row,column).
   *
   * @param row
   * @param column
   * @return double
   */
  double getValue(int row, int column) const;

  /**
   * @brief Generate random number between 0 and 1.
   *
   * @return double returns random number.
   */
  double generateRandomNumber();

  /**
   * @brief Prints all values in the metric.
   *
   */
  void printMatrixValues();

  /**
   * @brief Get the Number Of Columns in matrix.
   *
   * @return int number of columns.
   */
  int getNumberOfColumns() const;

  /**
   * @brief Get the number of rows in matrix.
   *
   * @return int number of rows.
   */
  int getNumberOfRows() const;

  /**
   * @brief Get the Matrix as row * columns.
   *
   * @return std::vector<std::vector<double>>  All values in matrix.
   */
  std::vector<std::vector<double>> getMatrix();

private:
  /** Number of rows in a matrix*/
  int m_numberOfRows;
  /** Number of columns in matrix. */
  int m_numberOfColumns;
  /** All matrix values rows times columns.*/
  std::vector<std::vector<double>> m_matrixValues;

public:
  /**
   * @brief Overload of multiplication operator.
   *
   * @param other reference to a Matrix object.
   * @return std::shared_ptr<Matrix> Pointer to the result Matrix of
   * multiplication.
   */
  std::shared_ptr<Matrix> operator*(const std::shared_ptr<Matrix> &other) const;

  /**
   * @brief Overload of the subtraction operator.
   *
   * @param other
   * @return std::shared_ptr<Matrix> Pointer to the result Matrix of
   * subtraction.
   */
  std::shared_ptr<Matrix> operator-(const std::shared_ptr<Matrix> &other) const;

  /**
   * @brief Overloaded multiplication operator for matrices using smart
   * pointers.
   *
   * Performs matrix multiplication using smart pointers.
   *
   * @param lhs The left-hand side matrix.
   * @param rhs The right-hand side matrix.
   * @return std::shared_ptr<Matrix> The result of the matrix multiplication.
   */
  friend std::shared_ptr<Matrix> operator*(const std::shared_ptr<Matrix> &lhs,
                                           const std::shared_ptr<Matrix> &rhs);

  /**
   * @brief Overloaded subtraction operator for matrices.
   *
   * Performs subtraction of two matrices.
   *
   * @param lhs The left-hand side matrix.
   * @param rhs The right-hand side matrix.
   * @return std::shared_ptr<Matrix> The result of the subtraction operation.
   */
  friend std::shared_ptr<Matrix> operator-(const std::shared_ptr<Matrix> &lhs,
                                           const std::shared_ptr<Matrix> &rhs);
};

#endif // _MATRIX_H