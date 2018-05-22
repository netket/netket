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

#ifndef NETKET_HAMILTONIAN_MATRIX_HPP
#define NETKET_HAMILTONIAN_MATRIX_HPP

#include <Eigen/Eigenvalues>
#include "Hilbert/hilbert_index.hpp"
#include "hamiltonian.hpp"

namespace netket {

class HamiltonianMatrix {
  const Hamiltonian& ham_;
  const Hilbert& hilbert_;
  HilbertIndex hilbert_index_;

 public:
  HamiltonianMatrix(const Hamiltonian& ham)
      : ham_(ham),
        hilbert_(ham.GetHilbert()),
        hilbert_index_(ham.GetHilbert()) {}

  Eigen::MatrixXcd Matrix() {
    auto nstates = hilbert_index_.NStates();
    Eigen::MatrixXcd h(nstates, nstates);

    h.setZero();

    std::vector<std::complex<double>> mel;
    std::vector<std::vector<int>> connectors;
    std::vector<std::vector<double>> newconfs;

    for (std::size_t i = 0; i < nstates; i++) {
      auto state = hilbert_index_.NumberToState(i);
      ham_.FindConn(state, mel, connectors, newconfs);

      for (std::size_t k = 0; k < mel.size(); k++) {
        auto state1 = state;
        hilbert_.UpdateConf(state1, connectors[k], newconfs[k]);
        auto j = hilbert_index_.StateToNumber(state1);

        h(i, j) += mel[k];
      }
    }
    return h;
  }

  Eigen::VectorXd EigenValues() {
    auto h = Matrix();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(h);
    return es.eigenvalues();
  }

  void SaveEigenValues(const std::string& filename, int first_n = 1) {
    std::ofstream file_ed(filename);

    auto eigs = EigenValues();

    eigs.conservativeResize(first_n);

    json j(eigs);
    file_ed << j << std::endl;

    file_ed.close();
  }
};
}  // namespace netket
#endif
