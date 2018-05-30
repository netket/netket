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
#include <fstream>
#include <iostream>
#include <limits>
#include "Utils/random_utils.hpp"

#include "hamiltonian_input_tests.hpp"
#include "netket.hpp"

TEST_CASE("hamiltonians produce elements in the hilbert space",
          "[hamiltonian]") {

  auto input_tests = GetHamiltonianInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {

    SECTION("Hamiltonian test (" + std::to_string(it) + ") on " +
            input_tests[it]["Hamiltonian"].dump()) {

      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      const netket::Hilbert &hilbert = hamiltonian.GetHilbert();

      const auto lstate = hilbert.LocalStates();
      REQUIRE(int(lstate.size()) == hilbert.LocalSize());

      std::set<int> lset(lstate.begin(), lstate.end());

      netket::default_random_engine rgen(3421);

      Eigen::VectorXd v(hilbert.Size());

      std::vector<std::complex<double>> mel;
      std::vector<std::vector<int>> connectors;
      std::vector<std::vector<double>> newconfs;

      for (int i = 0; i < 1000; i++) {
        hilbert.RandomVals(v, rgen);
        hamiltonian.FindConn(v, mel, connectors, newconfs);

        for (std::size_t k = 0; k < connectors.size(); k++) {
          Eigen::VectorXd vp = v;
          hilbert.UpdateConf(vp, connectors[k], newconfs[k]);
          for (int s = 0; s < vp.size(); s++) {
            REQUIRE(lset.count(vp(s)) > 0);
          }
        }
      }
    }
  }
}

TEST_CASE("hamiltonians do not have duplicate connections or newconfs", "[hamiltonian]") {

  auto input_tests = GetHamiltonianInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {

    SECTION("Hamiltonian test (" + std::to_string(it) + ") on " +
            input_tests[it]["Hamiltonian"].dump()) {

      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      const netket::Hilbert &hilbert = hamiltonian.GetHilbert();

      netket::default_random_engine rgen(3421);

      Eigen::VectorXd v(hilbert.Size());

      std::vector<std::complex<double>> mel;
      std::vector<std::vector<int>> connectors;
      std::vector<std::vector<double>> newconfs;

      for (int i = 0; i < 1000; i++) {
        hilbert.RandomVals(v, rgen);
        hamiltonian.FindConn(v, mel, connectors, newconfs);

        for (std::size_t k = 0; k < connectors.size(); k++) {
          if (connectors[k].size() > 0) {
            auto itu = std::unique(connectors[k].begin(), connectors[k].end());
            bool isUniqueConnector = (itu == connectors[k].end());
            REQUIRE(isUniqueConnector);
          }
        }

        auto itu = std::unique(connectors.begin(), connectors.end());
        bool noDuplicateConnectors = (itu == connectors.end());
        REQUIRE(noDuplicateConnectors);
      }
    }
  }
}

TEST_CASE("hamiltonians are hermitean", "[hamiltonian]") {

  auto input_tests = GetHamiltonianInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {

    SECTION("Hamiltonian test (" + std::to_string(it) + ") on " +
            input_tests[it]["Hamiltonian"].dump()) {

      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      const netket::Hilbert &hilbert = hamiltonian.GetHilbert();

      netket::default_random_engine rgen(3421);

      Eigen::VectorXd v(hilbert.Size());

      std::vector<std::complex<double>> mel;
      std::vector<std::vector<int>> connectors;
      std::vector<std::vector<double>> newconfs;

      for (int i = 0; i < 1000; i++) {
        hilbert.RandomVals(v, rgen);
        hamiltonian.FindConn(v, mel, connectors, newconfs);

        std::uniform_int_distribution<> dis(0, mel.size() - 1);
        int rc = dis(rgen);

        const auto melrc = mel[rc];

        Eigen::VectorXd vp = v;
        hilbert.UpdateConf(vp, connectors[rc], newconfs[rc]);
        hamiltonian.FindConn(vp, mel, connectors, newconfs);

        auto melic = melrc;
        bool found_inverse_mel = false;

        for (std::size_t k = 0; k < connectors.size(); k++) {
          Eigen::VectorXd vpp = vp;
          hilbert.UpdateConf(vpp, connectors[k], newconfs[k]);
          if (Approx((vpp - v).norm()) == 0) {
            melic = mel[k];
            found_inverse_mel = true;
            break;
          }
        }
        REQUIRE(found_inverse_mel == true);
        REQUIRE(Approx(std::real(melrc)) == std::real(std::conj(melic)));
        REQUIRE(Approx(std::imag(melrc)) == std::imag(std::conj(melic)));
      }
    }
  }
}
