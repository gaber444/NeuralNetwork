#include "matrix.h"

Matrix::Matrix(int numberOfRows, int numberOfColumns, bool isRandom)
    : m_numberOfRows(numberOfRows), m_numberOfColumns(numberOfColumns) {
  for (std::size_t row = 0; row < m_numberOfRows; row++) {
    std::vector<double> temporary;
    for (std::size_t column = 0; column < m_numberOfColumns; column++) {
      if (isRandom) {
        temporary.push_back(generateRandomNumber());
      } else {
        temporary.push_back(0.0);
      }
    }
    m_matrixValues.push_back(temporary);
  }
}

double Matrix::generateRandomNumber() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 1);

  return dis(gen);
}

void Matrix::printMatrixValues() {
  for (std::size_t row = 0; row < m_numberOfRows; ++row) {
    for (std::size_t col = 0; col < m_numberOfColumns; ++col) {
      std::cout << m_matrixValues.at(row).at(col);
      if (col < m_numberOfColumns - 1) {
        std::cout << ",";
      }
    }
    std::cout << std::endl;
  }
}

double Matrix::getValue(int row, int column) const {
  if (!m_matrixValues.empty()) {
    return m_matrixValues.at(row).at(column);
  } else {
    throw std::runtime_error("Matrix is empty.\n");
  }
}

int Matrix::getNumberOfColumns() const { return m_numberOfColumns; }

int Matrix::getNumberOfRows() const { return m_numberOfRows; }

void Matrix::setValue(int row, int column, double value) {
  if (!m_matrixValues.empty()) {
    m_matrixValues.at(row).at(column) = value;
  } else {
    throw std::runtime_error("Matrix is empty.\n");
  }
}

std::shared_ptr<Matrix> Matrix::transpose() {
  std::shared_ptr<Matrix> transposeMatrix =
      std::make_shared<Matrix>(m_numberOfColumns, m_numberOfRows, false);
  for (std::size_t r = 0; r < m_numberOfRows; ++r) {
    for (std::size_t c = 0; c < m_numberOfColumns; ++c) {
      transposeMatrix->setValue(c, r, getValue(r, c));
    }
  }
  return transposeMatrix;
}

std::vector<std::vector<double>> Matrix::getMatrix() { return m_matrixValues; }

std::shared_ptr<Matrix>
Matrix::operator*(const std::shared_ptr<Matrix> &other) const {

  if (m_numberOfColumns == 1 && m_numberOfRows == 1) {
    std::shared_ptr<Matrix> c = std::make_shared<Matrix>(
        other->m_numberOfRows, other->m_numberOfColumns, false);
    for (std::size_t row = 0; row < other->m_numberOfRows; ++row) {
      for (std::size_t col = 0; col < other->m_numberOfColumns; ++col) {
        c->setValue(row, col, getValue(0, 0) * other->getValue(row, col));
      }
    }
    return c;
  }
  if (other->m_numberOfColumns == 1 && other->m_numberOfRows == 1) {
    std::shared_ptr<Matrix> c =
        std::make_shared<Matrix>(m_numberOfRows, m_numberOfColumns, false);
    for (std::size_t row = 0; row < m_numberOfRows; ++row) {
      for (std::size_t col = 0; col < m_numberOfColumns; ++col) {
        c->setValue(row, col, getValue(row, col) * other->getValue(0, 0));
      }
    }
    return c;
  }
  if (m_numberOfColumns != other->m_numberOfRows) {
    std::cerr << "Matrix multiplication not possible.\n";
    return std::make_shared<Matrix>(0, 0, false);
  }

  // my result of matrix of multiplication
  std::shared_ptr<Matrix> c =
      std::make_shared<Matrix>(m_numberOfRows, other->m_numberOfColumns, false);

  for (std::size_t i = 0; i < m_numberOfRows; ++i) {
    for (std::size_t j = 0; j < other->m_numberOfColumns; ++j) {
      for (std::size_t k = 0; k < other->m_numberOfRows; ++k) {
        c->setValue(
            i, j, c->getValue(i, j) + (getValue(i, k) * other->getValue(k, j)));
      }
    }
  }
  return c;
}

// Non-member overloaded operator* for shared_ptr
std::shared_ptr<Matrix> operator*(const std::shared_ptr<Matrix> &lhs,
                                  const std::shared_ptr<Matrix> &rhs) {
  return (*lhs) * rhs;
}

std::shared_ptr<Matrix>
Matrix::operator-(const std::shared_ptr<Matrix> &other) const {
  if (m_numberOfRows != other->m_numberOfRows &&
      m_numberOfColumns != other->m_numberOfColumns) {
    std::cerr << "Matrix subtraction not possible.\n";
    return std::make_shared<Matrix>(0, 0, false);
  }
  // result of subtraction of two metrics.
  std::shared_ptr<Matrix> c =
      std::make_shared<Matrix>(m_numberOfRows, m_numberOfColumns, false);

  for (std::size_t row = 0; row < m_numberOfRows; ++row) {
    for (std::size_t column = 0; column < m_numberOfColumns; ++column) {
      c->setValue(row, column,
                  (getValue(row, column) - other->getValue(row, column)));
    }
  }
  return c;
}

std::shared_ptr<Matrix> operator-(const std::shared_ptr<Matrix> &lhs,
                                  const std::shared_ptr<Matrix> &rhs) {
  return (*lhs) - rhs;
}