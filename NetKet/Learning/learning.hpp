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

#include "Hamiltonian/MatrixWrapper/dense_matrix_wrapper.hpp"
#include "Optimizer/optimizer.hpp"
#include "ground_state.hpp"

namespace netket {

class Learning {
public:
  explicit Learning(const json &pars) {
    CheckFieldExists(pars, "Learning");
    const std::string method_name =
        FieldVal(pars["Learning"], "Method", "Learning");

    if (method_name == "Gd" || method_name == "Sr") {
      Graph graph(pars);

      Hamiltonian hamiltonian(graph, pars);

      using MachineType = Machine<std::complex<double>>;
      MachineType machine(graph, hamiltonian, pars);

      Sampler<MachineType> sampler(graph, hamiltonian, machine, pars);

      Optimizer optimizer(pars);

      GroundState le(hamiltonian, sampler, optimizer, pars);
    } else if (method_name == "Ed") {
      Graph graph(pars);

      Hamiltonian hamiltonian(graph, pars);
      std::string file_base = FieldVal(pars["Learning"], "OutputFile");

      SaveEigenValues(hamiltonian, file_base + std::string(".log"));

    } else {
      std::stringstream s;
      s << "Unknown Learning method: " << method_name;
      throw InvalidInputError(s.str());
    }
  }

  void SaveEigenValues(const Hamiltonian &hamiltonian,
                       const std::string &filename, int first_n = 1) {
    std::ofstream file_ed(filename);

    auto matrix = DenseMatrixWrapper<Hamiltonian>(hamiltonian);
    auto ed = matrix.ComputeEigendecomposition(Eigen::EigenvaluesOnly);

    auto eigs = ed.eigenvalues();
    eigs.conservativeResize(first_n);

    json j(eigs);
    file_ed << j << std::endl;

    file_ed.close();
  }
};

} // namespace netket

#endif
