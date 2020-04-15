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

#ifndef NETKET_ARRAYHASHER_HPP
#define NETKET_ARRAYHASHER_HPP

#include <array>
#include "common_types.hpp"

namespace netket {
// Special hash functor for the EdgeColors unordered_map
// Same as hash_combine from boost
struct ArrayHasher {
  std::size_t operator()(const std::array<int, 2>& a) const noexcept {
    return *reinterpret_cast<std::size_t const*>(a.data());
  }
};

template <typename T>
struct EigenArrayHasher {
  std::size_t operator()(const T& array) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column-
    // or row-major). It will give you the same hash value for two different
    // matrices if they are the transpose of each other in different storage
    // order.
    size_t seed = 0;
    for (Index i = 0; i < array.size(); ++i) {
      auto elem = array(i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

template <typename T>
struct EigenArrayEqualityComparison {
  bool operator()(const T& a, const T& b) const { return a.isApprox(b); }
};

}  // namespace netket
#endif
