// Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "set.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace netket {

Set::Set(int size) : n_sites_(size) {
  if (size < 0) {
    throw InvalidInputError{"Invalid size provided in Set."};
  }

  automorphisms_.resize(1);
  automorphisms_.front().resize(static_cast<std::size_t>(n_sites_));
  std::iota(std::begin(automorphisms_.front()),
            std::end(automorphisms_.front()), 0);
}

int Set::Nsites() const noexcept { return n_sites_; }

int Set::Size() const noexcept { return n_sites_; }

bool Set::IsConnected() const noexcept { return false; }

bool Set::IsBipartite() const noexcept { return false; }

std::vector<Set::Edge> const &Set::Edges() const noexcept { return edges_; }

std::vector<std::vector<int>> Set::AdjacencyList() const {
  return std::vector<std::vector<int>>(n_sites_);
}

const Set::ColorMap &Set::EdgeColors() const noexcept { return cmap_; }

// Returns a list of permuted sites constituting an automorphism of the
// graph
std::vector<std::vector<int>> Set::SymmetryTable() const {
  return automorphisms_;
}

}  // namespace netket
