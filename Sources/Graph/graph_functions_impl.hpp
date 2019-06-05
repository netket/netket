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

#ifndef NETKET_GRAPH_FUNCTIONS_IMPL_HPP
#define NETKET_GRAPH_FUNCTIONS_IMPL_HPP

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

std::vector<int> AbstractGraph::Distances(int root) const {
  std::vector<int> dists(Nsites(), -1);

  // Dijkstra's algorithm
  BreadthFirstSearch(root,
                     [&dists](int node, int depth) { dists[node] = depth; });

  return dists;
}

std::vector<std::vector<int>> AbstractGraph::AllDistances() const {
  // This can be implemented more efficiently (by reusing information between
  // calls to Distances) if necessary.
  std::vector<std::vector<int>> distances;
  for (int i = 0; i < Nsites(); i++) {
    distances.push_back(Distances(i));
  }
  return distances;
}
bool AbstractGraph::IsConnected() const noexcept {
  const int start = 0;  // arbitrary node
  int nvisited = 0;
  BreadthFirstSearch(start, [&nvisited](int, int) { ++nvisited; });
  return nvisited == Nsites();
}

bool AbstractGraph::IsBipartite() const noexcept {
  bool is_bipartite = true;
  const int start = 0;  // arbitrary node
  std::vector<int> colors(Nsites(), -1);
  const auto adjacency_list =
      AdjacencyList();  // implicit expression can't have
                        // access to the Lattice function
  if (IsConnected()) {
    BreadthFirstSearch(
        start, [&colors, &adjacency_list, &is_bipartite](int node, int) {
          if (node == start) colors[node] = 1;
          for (std::size_t j = 0; j < adjacency_list[node].size(); j++) {
            if (!is_bipartite) break;
            if (colors[adjacency_list[node][j]] == -1) {
              colors[adjacency_list[node][j]] = 1 - colors[node];
            } else if (colors[adjacency_list[node][j]] == colors[node]) {
              is_bipartite = false;
            }
          }
        });
  } else {
    BreadthFirstSearch(
        [&colors, &adjacency_list, &is_bipartite](int node, int, int) {
          if (node == start) colors[node] = 1;
          for (std::size_t j = 0; j < adjacency_list[node].size(); j++) {
            if (!is_bipartite) break;
            if (colors[adjacency_list[node][j]] == -1) {
              colors[adjacency_list[node][j]] = 1 - colors[node];
            } else if (colors[adjacency_list[node][j]] == colors[node]) {
              is_bipartite = false;
            }
          }
        });
  }
  return is_bipartite;
}

}  // namespace netket
#endif
