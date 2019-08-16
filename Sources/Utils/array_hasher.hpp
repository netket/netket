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

namespace netket {
// Special hash functor for the EdgeColors unordered_map
// Same as hash_combine from boost
struct ArrayHasher {
  std::size_t operator()(const std::array<int, 2>& a) const noexcept {
    return *reinterpret_cast<std::size_t const*>(a.data());
  }
};
}  // namespace netket

#endif
