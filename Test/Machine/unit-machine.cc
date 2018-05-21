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

#include <fstream>
#include <iostream>
#include <limits>
#include "catch.hpp"

#include "machine_input_tests.hpp"
#include "netket.hpp"

TEST_CASE("machines set/get correctly parameters", "[machine]") {
  auto input_tests = GetMachineInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Machine test (" + std::to_string(it) + ") on " +
            input_tests[it]["Machine"].dump()) {
      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      using MType = std::complex<double>;

      netket::Machine<MType> machine(graph, hamiltonian, pars);

      int seed = 12342;
      double sigma = 1;
      netket::Machine<MType>::VectorType params(machine.Npar());
      netket::RandomGaussian(params, seed, sigma);

      machine.SetParameters(params);

      REQUIRE(Approx((machine.GetParameters() - params).norm()) == 0);
    }
  }
}

TEST_CASE("machines compute log derivatives correctly", "[machine]") {
  auto input_tests = GetMachineInputs();
  std::size_t ntests = input_tests.size();

  netket::default_random_engine rgen;

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Machine test (" + std::to_string(it) + ") on " +
            input_tests[it]["Machine"].dump()) {
      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      using MType = std::complex<double>;

      netket::Machine<MType> machine(graph, hamiltonian, pars);

      double sigma = 1.;
      machine.InitRandomPars(1234, sigma);

      const netket::Hilbert &hilbert = hamiltonian.GetHilbert();

      int nv = hilbert.Size();
      Eigen::VectorXd v(nv);

      double eps = std::sqrt(std::numeric_limits<double>::epsilon()) * 1000;

      for (int i = 0; i < 100; i++) {
        hilbert.RandomVals(v, rgen);

        auto ders = machine.DerLog(v);

        auto machine_pars = machine.GetParameters();

        for (int p = 0; p < machine.Npar(); p++) {
          machine_pars(p) += eps;
          machine.SetParameters(machine_pars);
          typename netket::Machine<MType>::StateType valp = machine.LogVal(v);

          machine_pars(p) -= 2 * eps;
          machine.SetParameters(machine_pars);
          typename netket::Machine<MType>::StateType valm = machine.LogVal(v);

          machine_pars(p) += eps;

          typename netket::Machine<MType>::StateType numder =
              (-valm + valp) / (eps * 2);

          REQUIRE(Approx(std::real(numder)).epsilon(eps * 100) ==
                  std::real(ders(p)));
          REQUIRE(Approx(std::exp(std::imag(numder))).epsilon(eps * 100) ==
                  std::exp(std::imag(ders(p))));
        }
      }
    }
  }
}

TEST_CASE("machines compute logval differences correctly", "[machine]") {
  auto input_tests = GetMachineInputs();
  std::size_t ntests = input_tests.size();

  netket::default_random_engine rgen;

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Machine test (" + std::to_string(it) + ") on " +
            input_tests[it]["Machine"].dump()) {
      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      using MType = std::complex<double>;
      using WfType = netket::Machine<MType>;

      WfType machine(graph, hamiltonian, pars);

      double sigma = 1;
      machine.InitRandomPars(1234, sigma);

      const netket::Hilbert &hilbert = hamiltonian.GetHilbert();

      typename WfType::LookupType lt;

      int nv = hilbert.Size();
      Eigen::VectorXd v(nv);

      int nstates = hilbert.LocalSize();
      const auto localstates = hilbert.LocalStates();

      std::uniform_int_distribution<int> diststate(0, nstates - 1);
      std::uniform_int_distribution<int> distnchange(0, nv - 1);

      std::vector<int> randperm(nv);
      for (int i = 0; i < nv; i++) {
        randperm[i] = i;
      }

      for (int i = 0; i < 100; i++) {
        hilbert.RandomVals(v, rgen);
        machine.InitLookup(v, lt);

        auto valold = machine.LogVal(v);

        // we test on a random number of sites to be changed
        int nchange = distnchange(rgen);
        std::vector<int> tochange(nchange);
        std::vector<double> newconf(nchange);

        // picking k unique random sites to be changed
        std::random_shuffle(randperm.begin(), randperm.end());

        for (int k = 0; k < nchange; k++) {
          int si = randperm[k];

          tochange[k] = si;

          // picking a random state
          int newstate = diststate(rgen);
          newconf[k] = localstates[newstate];
        }

        const auto lvd = machine.LogValDiff(v, tochange, newconf, lt);

        if (nchange > 0) {
          hilbert.UpdateConf(v, tochange, newconf);
          auto valnew = machine.LogVal(v);

          REQUIRE(Approx(std::real(std::exp(lvd))).margin(1.0e-6) ==
                  std::real(std::exp(valnew - valold)));
          REQUIRE(Approx(std::imag(std::exp(lvd))).margin(1.0e-6) ==
                  std::imag(std::exp(valnew - valold)));
        } else {
          REQUIRE(Approx(std::real(std::exp(lvd))) == 1.0);
          REQUIRE(Approx(std::imag(std::exp(lvd))) == 0.0);
        }
      }
    }
  }
}

TEST_CASE("machines update look-up tables correctly", "[machine]") {
  auto input_tests = GetMachineInputs();
  std::size_t ntests = input_tests.size();

  netket::default_random_engine rgen;

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Machine test (" + std::to_string(it) + ") on " +
            input_tests[it]["Machine"].dump()) {
      auto pars = input_tests[it];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      using MType = std::complex<double>;
      using WfType = netket::Machine<MType>;

      WfType machine(graph, hamiltonian, pars);

      double sigma = 1;
      machine.InitRandomPars(1234, sigma);

      const netket::Hilbert &hilbert = hamiltonian.GetHilbert();

      typename WfType::LookupType lt;
      typename WfType::LookupType ltnew;

      int nv = hilbert.Size();
      Eigen::VectorXd v(nv);

      int nstates = hilbert.LocalSize();
      const auto localstates = hilbert.LocalStates();

      std::uniform_int_distribution<int> diststate(0, nstates - 1);
      std::uniform_int_distribution<int> distnchange(0, nv - 1);

      std::vector<int> randperm(nv);
      for (int i = 0; i < nv; i++) {
        randperm[i] = i;
      }

      hilbert.RandomVals(v, rgen);
      machine.InitLookup(v, lt);

      for (int i = 0; i < 100; i++) {
        // we test on a random number of sites to be changed
        int nchange = distnchange(rgen);
        std::vector<int> tochange(nchange);
        std::vector<double> newconf(nchange);

        // picking k unique random site to be changed
        std::random_shuffle(randperm.begin(), randperm.end());

        for (int k = 0; k < nchange; k++) {
          int si = randperm[k];

          tochange[k] = si;

          // picking a random state
          int newstate = diststate(rgen);
          newconf[k] = localstates[newstate];
        }

        machine.UpdateLookup(v, tochange, newconf, lt);
        hilbert.UpdateConf(v, tochange, newconf);

        machine.InitLookup(v, ltnew);

        for (int vlt = 0; vlt < lt.VectorSize(); vlt++) {
          for (int k = 0; k < lt.V(vlt).size(); k++) {
            REQUIRE(Approx(std::real(lt.V(vlt)(k))).margin(1.0e-6) ==
                    std::real(ltnew.V(vlt)(k)));
            REQUIRE(Approx(std::imag(lt.V(vlt)(k))).margin(1.0e-6) ==
                    std::imag(ltnew.V(vlt)(k)));
          }
        }

        for (int mlt = 0; mlt < lt.MatrixSize(); mlt++) {
          for (int k = 0; k < lt.M(mlt).rows(); k++) {
            for (int kp = 0; kp < lt.M(mlt).cols(); kp++) {
              REQUIRE(Approx(std::real(lt.M(mlt)(k, kp))).margin(1.0e-6) ==
                      std::real(ltnew.M(mlt)(k, kp)));
              REQUIRE(Approx(std::imag(lt.M(mlt)(k, kp))).margin(1.0e-6) ==
                      std::imag(ltnew.M(mlt)(k, kp)));
            }
          }
        }
      }
    }
  }
}
