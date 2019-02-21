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

#include "layer_input_tests.hpp"
#include "netket.hpp"

TEST_CASE("layers set/get correctly parameters", "[layers]") {
  auto input_tests = GetLayerInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Layer test (" + std::to_string(it) + ") on " +
            input_tests[it]["Machine"]["Layers"].dump()) {
      auto pars = input_tests[it];

      using MType = Complex;
      netket::Graph graph(pars);
      netket::Hilbert hilbert(graph, pars);

      REQUIRE(netket::FieldExists(pars, "Machine"));
      REQUIRE(netket::FieldExists(pars["Machine"], "Layers"));
      REQUIRE(pars["Machine"]["Layers"].size() > 0);
      std::cout << pars["Machine"]["Layers"][0] << std::endl;
      netket::Layer<MType> layer(graph, pars["Machine"]["Layers"][0]);

      int seed = 12342;
      double sigma = 1;
      std::cout << layer.Npar() << std::endl;

      // BUG Check segmentation fault here
      netket::Layer<MType>::VectorType params_in(layer.Npar());
      netket::Layer<MType>::VectorType params_out(layer.Npar());
      netket::RandomGaussian(params_in, seed, sigma);

      layer.SetParameters(params_in, 0);
      layer.GetParameters(params_out, 0);

      REQUIRE(Approx((params_out - params_in).norm()) == 0);
    }
  }
}
//
// TEST_CASE("machines write/read to/from json correctly", "[layers]") {
//   auto input_tests = GetLayerInputs();
//   std::size_t ntests = input_tests.size();
//
//   for (std::size_t it = 0; it < ntests; it++) {
//     SECTION("Layer test (" + std::to_string(it) + ") on " +
//             input_tests[it]["Machine"].dump()) {
//       auto pars = input_tests[it];
//
//       netket::Graph graph(pars);
//       netket::Hilbert hilbert(graph, pars);
//
//       netket::Hamiltonian hamiltonian(hilbert, pars);
//
//       using MType = Complex;
//
//       netket::Machine<MType> machine(graph, hamiltonian, pars);
//
//       int seed = 12342;
//       double sigma = 1;
//       netket::Machine<MType>::VectorType params(machine.Npar());
//       netket::RandomGaussian(params, seed, sigma);
//
//       machine.SetParameters(params);
//
//       netket::json pars_out;
//       machine.to_json(pars_out);
//
//       machine.from_json(pars_out["Machine"]);
//
//       netket::Machine<MType>::VectorType params_out(machine.Npar());
//
//       params_out = machine.GetParameters();
//
//       REQUIRE(Approx((params_out - params).norm()) == 0);
//     }
//   }
// }
//
// TEST_CASE("layers compute log derivatives correctly", "[layer]") {
//   auto input_tests = GetLayerInputs();
//   std::size_t ntests = input_tests.size();
//
//   netket::default_random_engine rgen;
//
//   for (std::size_t it = 0; it < ntests; it++) {
//     SECTION("Layer test (" + std::to_string(it) + ") on " +
//             input_tests[it]["Machine"].dump()) {
//       auto pars = input_tests[it];
//
//       netket::Graph graph(pars);
//       netket::Hilbert hilbert(graph, pars);
//
//       netket::Hamiltonian hamiltonian(hilbert, pars);
//
//       using MType = Complex;
//
//       netket::Machine<MType> machine(graph, hamiltonian, pars);
//
//       double sigma = 1.;
//       machine.InitRandomPars(1234, sigma);
//
//       int nv = hilbert.Size();
//       Eigen::VectorXd v(nv);
//
//       double eps = std::sqrt(std::numeric_limits<double>::epsilon()) * 100;
//       for (int i = 0; i < 100; i++) {
//         hilbert.RandomVals(v, rgen);
//
//         auto ders = machine.DerLog(v);
//
//         auto machine_pars = machine.GetParameters();
//
//         for (int p = 0; p < machine.Npar(); p++) {
//           machine_pars(p) += eps;
//           machine.SetParameters(machine_pars);
//           typename netket::Machine<MType>::StateType valp =
//           machine.LogVal(v);
//
//           machine_pars(p) -= 2 * eps;
//           machine.SetParameters(machine_pars);
//           typename netket::Machine<MType>::StateType valm =
//           machine.LogVal(v);
//
//           machine_pars(p) += eps;
//
//           typename netket::Machine<MType>::StateType numder =
//               (-valm + valp) / (eps * 2);
//
//           REQUIRE(Approx(std::real(numder)).epsilon(eps * 1000) ==
//                   std::real(ders(p)));
//           REQUIRE(Approx(std::exp(std::imag(numder))).epsilon(eps * 1000) ==
//                   std::exp(std::imag(ders(p))));
//         }
//       }
//     }
//   }
// }

// TEST_CASE("Layers update look-up tables correctly", "[layer]") {
//   auto input_tests = GetLayerInputs();
//   std::size_t ntests = input_tests.size();
//
//   netket::default_random_engine rgen;
//
//   for (std::size_t it = 0; it < ntests; it++) {
//     SECTION("Layer test (" + std::to_string(it) + ") on " +
//             input_tests[it]["Machine"].dump()) {
//       auto pars = input_tests[it];
//
//       netket::Graph graph(pars);
//       netket::Hilbert hilbert(graph, pars);
//
//       netket::Hamiltonian hamiltonian(hilbert, pars);
//
//       using MType = Complex;
//       using WfType = netket::Machine<MType>;
//
//       WfType machine(graph, hamiltonian, pars);
//
//       double sigma = 1;
//       machine.InitRandomPars(1234, sigma);
//
//       typename WfType::LookupType lt;
//       typename WfType::LookupType ltnew;
//
//       int nv = hilbert.Size();
//       Eigen::VectorXd v(nv);
//
//       int nstates = hilbert.LocalSize();
//       const auto localstates = hilbert.LocalStates();
//
//       std::uniform_int_distribution<int> diststate(0, nstates - 1);
//       std::uniform_int_distribution<int> distnchange(0, nv - 1);
//
//       std::vector<int> randperm(nv);
//       for (int i = 0; i < nv; i++) {
//         randperm[i] = i;
//       }
//
//       hilbert.RandomVals(v, rgen);
//       machine.InitLookup(v, lt);
//
//       for (int i = 0; i < 100; i++) {
//         // we test on a random number of sites to be changed
//         int nchange = distnchange(rgen);
//         std::vector<int> tochange(nchange);
//         std::vector<double> newconf(nchange);
//
//         // picking k unique random site to be changed
//         std::random_shuffle(randperm.begin(), randperm.end());
//
//         for (int k = 0; k < nchange; k++) {
//           int si = randperm[k];
//
//           tochange[k] = si;
//
//           // picking a random state
//           int newstate = diststate(rgen);
//           newconf[k] = localstates[newstate];
//         }
//
//         machine.UpdateLookup(v, tochange, newconf, lt);
//         hilbert.UpdateConf(v, tochange, newconf);
//
//         machine.InitLookup(v, ltnew);
//
//         for (int vlt = 0; vlt < lt.VectorSize(); vlt++) {
//           for (int k = 0; k < lt.V(vlt).size(); k++) {
//             REQUIRE(Approx(std::real(lt.V(vlt)(k))).margin(1.0e-6) ==
//                     std::real(ltnew.V(vlt)(k)));
//             REQUIRE(Approx(std::imag(lt.V(vlt)(k))).margin(1.0e-6) ==
//                     std::imag(ltnew.V(vlt)(k)));
//           }
//         }
//
//         for (int mlt = 0; mlt < lt.VVSize(); mlt++) {
//           for (int k = 0; k < int(lt.VV(mlt).size()); k++) {
//             for (int kp = 0; kp < int(lt.VV(mlt)[k].size()); kp++) {
//               REQUIRE(Approx(std::real(lt.VV(mlt)[k](kp))).margin(1.0e-6) ==
//                       std::real(ltnew.VV(mlt)[k](kp)));
//               REQUIRE(Approx(std::imag(lt.VV(mlt)[k](kp))).margin(1.0e-6) ==
//                       std::imag(ltnew.VV(mlt)[k](kp)));
//             }
//           }
//         }
//       }
//     }
//   }
// }
