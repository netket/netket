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

#include <cassert>
#include <queue>

#include "Utils/exceptions.hpp"

namespace netket {

template <typename Func>
void AbstractGraph::BreadthFirstSearch_Impl(int start, int max_depth,
                                            Func visitor_func,
                                            std::vector<bool> &seen) const {
  assert(start >= 0 && start < Nsites());
  assert(max_depth > 0);
  assert(std::ptrdiff_t(seen.size()) == Nsites());
  assert(!seen[start]);

  // Queue to store states to visit
  using QueueEntry = std::pair<int, int>;
  std::queue<QueueEntry> queue;
  queue.push({start, 0});

  seen[start] = true;

  const auto adjacency_list = AdjacencyList();
  while (!queue.empty()) {
    const auto elem = queue.front();
    queue.pop();
    const int node = elem.first;
    const int depth = elem.second;

    if (depth > max_depth) {
      continue;
    }

    visitor_func(node, depth);

    for (const int adj : adjacency_list[node]) {
      if (!seen[adj]) {
        seen[adj] = true;
        queue.push({adj, depth + 1});
      }
    }
  }
}

template <typename Func>
void AbstractGraph::BreadthFirstSearch(Func visitor_func) const {
  std::vector<bool> seen(Nsites(), false);
  for (int v = 0; v < Nsites(); ++v) {
    if (seen[v]) {
      continue;
    }
    auto modified_visitor = [&](int node, int depth) {
      visitor_func(node, depth, v);
    };
    BreadthFirstSearch_Impl(v, Nsites(), modified_visitor, seen);
  }
}

template <typename Func>
void AbstractGraph::BreadthFirstSearch(int start, int max_depth,
                                       Func visitor_func) const {
  std::vector<bool> seen(Nsites(), false);
  BreadthFirstSearch_Impl(start, max_depth, visitor_func, seen);
}

template <typename Func>
void AbstractGraph::BreadthFirstSearch(int start, Func visitor_func) const {
BreadthFirstSearch(start, Nsites(), visitor_func);
}

}  // namespace netket
