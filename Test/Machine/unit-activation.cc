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
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"
#include "catch.hpp"

#include "netket.hpp"

netket::json GetActivationInputs() {
  netket::json pars;

  // Activations
  pars = {{"Activations",
           {{{"Activation", "Lncosh"}},
            {{"Activation", "Identity"}},
            {{"Activation", "Tanh"}}}}};

  return pars;
}

TEST_CASE("activations compute derivatives correctly", "[activation]") {
  auto input_tests = GetActivationInputs();
  std::size_t ntests = input_tests["Activations"].size();

  netket::default_random_engine rgen;

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Activation test (" + std::to_string(it) + ") on " +
            input_tests["Activations"][it].dump()) {
      auto pars = input_tests["Activations"][it];

      netket::Activation activation(pars);

      using MType = Complex;

      double eps = std::sqrt(std::numeric_limits<double>::epsilon()) * 1000;

      netket::Layer<MType>::VectorType eps_vec(100);
      netket::Layer<MType>::VectorType test_var(100);
      netket::Layer<MType>::VectorType test_varp(100);
      netket::Layer<MType>::VectorType test_varm(100);
      netket::Layer<MType>::VectorType val(100);
      netket::Layer<MType>::VectorType valp(100);
      netket::Layer<MType>::VectorType valm(100);
      netket::Layer<MType>::VectorType check_der(100);
      netket::Layer<MType>::VectorType comp_der(100);
      netket::Layer<MType>::VectorType dLdA(100);

      netket::RandomGaussian(test_var, 1232, 0.5);
      eps_vec.setConstant(eps);
      dLdA.setConstant(1);

      test_varp = test_var + eps_vec;
      test_varm = test_var - eps_vec;

      activation(test_varp, valp);
      activation(test_varm, valm);
      activation(test_var, val);

      check_der = (valp - valm) / (eps * 2);
      activation.ApplyJacobian(test_var, val, dLdA, comp_der);

      for (int p = 0; p < 100; p++) {
        REQUIRE(Approx(std::real(check_der(p))).epsilon(eps * 100) ==
                std::real(comp_der(p)));
        REQUIRE(Approx(std::exp(std::imag(check_der(p)))).epsilon(eps * 100) ==
                std::exp(std::imag(comp_der(p))));
      }
    }
  }
}
