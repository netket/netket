//
// Created by Filippo Vicentini on 2019-06-06.
//

#ifndef NETKET_ARRAY_UTILS_HPP
#define NETKET_ARRAY_UTILS_HPP

#include <algorithm>
#include <complex>
#include <vector>

namespace netket {

template <typename T>
std::vector<std::vector<T>> transpose_vecvec(
    const std::vector<std::vector<T>>& data) {
  typedef typename std::vector<T>::size_type size_type;

  // this assumes that all inner vectors have the same size and
  // allocates space for the complete result in advance
  std::vector<std::vector<T>> result(data[0].size(),
                                     std::vector<T>(data.size()));
  for (size_type i = 0; i < data[0].size(); i++)
    for (size_type j = 0; j < data.size(); j++) {
      result[i][j] = data[j][i];
    }
  return result;
}

}  // namespace netket
#endif  // NETKET_ARRAY_UTILS_HPP
