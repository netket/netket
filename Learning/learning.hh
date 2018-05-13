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

#ifndef NETKET_LEARNING_CC
#define NETKET_LEARNING_CC

#include <memory>

#include "ground_state.hh"
#include "stepper.hh"

namespace netket {

class Learning {

public:
  Learning(const json &pars) {

    if (!FieldExists(pars, "Learning")) {
      std::cerr << "Learning field is not defined in the input" << std::endl;
      std::abort();
    }

    if (!FieldExists(pars["Learning"], "Method")) {
      std::cerr << "Learning Method is not defined in the input" << std::endl;
      std::abort();
    }

    if (pars["Learning"]["Method"] == "Gd" ||
        pars["Learning"]["Method"] == "Sr") {

      Graph graph(pars);

      Hamiltonian hamiltonian(graph, pars);

      using MachineType = Machine<std::complex<double>>;
      MachineType machine(graph, hamiltonian, pars);

      Sampler<MachineType> sampler(graph, hamiltonian, machine, pars);

      Stepper stepper(pars);

      GroundState le(hamiltonian, sampler, stepper, pars);
    } else {
      std::cout << "Learning method not found" << std::endl;
      std::cout << pars["Learning"]["Method"] << std::endl;
      std::abort();
    }
  }
};

} // namespace netket

#endif
