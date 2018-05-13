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

#ifndef NETKET_FINDDIST_HH
#define NETKET_FINDDIST_HH

#include <queue>
#include <set>
#include <vector>

namespace netket {

std::vector<int> FindDist(const std::vector<std::vector<int>> &g, int root) {
  int n = g.size();
  std::vector<int> dists(n, -1);

  dists[root] = 0;

  std::queue<int> tovisit;

  tovisit.push(root);

  while (tovisit.size() > 0) {
    int node = tovisit.front();
    tovisit.pop();

    for (std::size_t j = 0; j < g[node].size(); j++) {
      int nj = g[node][j];

      if (dists[nj] == -1) {
        tovisit.push(nj);
        dists[nj] = dists[node] + 1;
      }
    }
  }

  return dists;
}

} // namespace netket
#endif
