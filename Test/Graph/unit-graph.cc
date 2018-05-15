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

#include "catch.hpp"
#include "netket.hh"
#include <fstream>
#include <iostream>
#include <vector>

#include "graph_input_tests.hh"

TEST_CASE("graphs have consistent number of sites", "[graph]") {

  auto input_tests = GetGraphInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i].dump();

    SECTION("Graph test on " + name) {

      netket::Graph graph(input_tests[i]);

      REQUIRE(graph.Nsites() > 0);
    }
  }
}
