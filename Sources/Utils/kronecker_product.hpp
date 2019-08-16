// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_KRONECKER_PRODUCT_HPP
#define NETKET_KRONECKER_PRODUCT_HPP

#include <vector>

namespace netket {

template <class T>
bool isMatrix(const std::vector<std::vector<T>>& Am) {
  const std::size_t rowA = Am.size();
  if (rowA == 0) {
    return true;
  }
  const std::size_t colA = Am[0].size();
  for (std::size_t i = 0; i < rowA; i++) {
    if (Am[i].size() != colA) {
      return false;
    }
  }
  return true;
}

template <class T>
std::vector<std::vector<T>> KroneckerProduct(
    const std::vector<std::vector<T>>& Am,
    const std::vector<std::vector<T>>& Bm) {
  const std::size_t rowA = Am.size();
  const std::size_t rowB = Bm.size();

  assert(rowA > 0);
  assert(rowB > 0);
  assert(isMatrix(Am));
  assert(isMatrix(Bm));

  const std::size_t colA = Am[0].size();
  const std::size_t colB = Bm[0].size();

  const std::size_t rowC = rowA * rowB;
  const std::size_t colC = colA * colB;
  std::vector<std::vector<T>> Cm(rowC, std::vector<T>(colC, 0.));

  for (std::size_t i = 0; i < rowA; i++) {
    for (std::size_t j = 0; j < colA; j++) {
      std::size_t startRow = i * rowB;
      std::size_t startCol = j * colB;
      for (std::size_t k = 0; k < rowB; k++) {
        for (std::size_t l = 0; l < colB; l++) {
          Cm[startRow + k][startCol + l] = Am[i][j] * Bm[k][l];
        }
      }
    }
  }
  return Cm;
}

template <class T>
std::vector<std::vector<T>> MatrixProduct(
    const std::vector<std::vector<T>>& Am,
    const std::vector<std::vector<T>>& Bm) {
  const std::size_t rowA = Am.size();
#ifndef NDEBUG
  const std::size_t rowB = Bm.size();
#endif
  assert(rowA > 0);
  assert(rowB > 0);
  assert(isMatrix(Am));
  assert(isMatrix(Bm));

  const std::size_t colA = Am[0].size();
  const std::size_t colB = Bm[0].size();

  assert(colA == rowB);

  const std::size_t rowC = rowA;
  const std::size_t colC = colB;
  std::vector<std::vector<T>> Cm(rowC, std::vector<T>(colC, 0.));

  for (std::size_t i = 0; i < rowC; i++) {
    for (std::size_t j = 0; j < colC; j++) {
      for (std::size_t k = 0; k < colA; k++) {
        Cm[i][j] += Am[i][k] * Bm[k][j];
      }
    }
  }

  return Cm;
}

}  // namespace netket

#endif
