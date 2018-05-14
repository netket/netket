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
#include <cfloat>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include "hilbert_input_tests.hh"

TEST_CASE("hilbert has consistent sizes and definitions", "[hilbert]") {

  auto input_tests = GetHilbertInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i]["Hilbert"].dump();

    SECTION("Hilbert test on " + name) {

      netket::Hilbert hilbert(input_tests[i]);

      REQUIRE(hilbert.Size() > 0);
      REQUIRE(hilbert.LocalSize() > 0);

      if (hilbert.IsDiscrete()) {
        const auto lstate = hilbert.LocalStates();

        REQUIRE(int(lstate.size()) == hilbert.LocalSize());

        for (std::size_t k = 0; k < lstate.size(); k++) {
          REQUIRE(std::isfinite(lstate[k]));
        }
      }
    }
  }
}

TEST_CASE("hilbert generates consistent random states", "[hilbert]") {

  auto input_tests = GetHilbertInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string parname = "Hilbert";
    if (!netket::FieldExists(input_tests[i], "Hilbert")) {
      parname = "Hamiltonian";
    }
    std::string name = input_tests[i][parname].dump();

    SECTION("Hilbert test on " + name) {

      netket::Hilbert hilbert;

      if (netket::FieldExists(input_tests[i], "Hilbert")) {
        hilbert = netket::Hilbert(input_tests[i]);
      } else if (netket::FieldExists(input_tests[i], "Hamiltonian")) {
        netket::Graph graph(input_tests[i]);
        netket::Hamiltonian hamiltonian(graph, input_tests[i]);
        hilbert = netket::Hilbert(hamiltonian.GetHilbert());
      }

      REQUIRE(hilbert.Size() > 0);

      if (hilbert.IsDiscrete()) {

        netket::default_random_engine rgen(3421);
        Eigen::VectorXd rstate(hilbert.Size());

        const auto lstate = hilbert.LocalStates();
        REQUIRE(int(lstate.size()) == hilbert.LocalSize());

        std::set<int> lset(lstate.begin(), lstate.end());

        for (int it = 0; it < 100; it++) {
          hilbert.RandomVals(rstate, rgen);
          for (int k = 0; k < rstate.size(); k++) {
            REQUIRE(lset.count(rstate(k)) > 0);
          }
        }
      }
    }
  }
}
