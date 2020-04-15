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

#include "hamiltonian_kernel.hpp"

namespace netket {

HamiltonianKernel::HamiltonianKernel(const AbstractMachine &psi,
                                     AbstractOperator &hamiltonian)
    : hamiltonian_(hamiltonian), nv_(psi.GetHilbert().Size()) {
  NETKET_CHECK(psi.GetHilbert().IsDiscrete(), InvalidInputError,
               "Hamiltonian Metropolis sampler works only for discrete "
               "Hilbert spaces");
}

HamiltonianKernel::HamiltonianKernel(AbstractOperator &ham)
    : hamiltonian_(ham), nv_(ham.GetHilbert().Size()) {
  NETKET_CHECK(ham.GetHilbert().IsDiscrete(), InvalidInputError,
               "Hamiltonian Metropolis sampler works only for discrete "
               "Hilbert spaces");
}

void HamiltonianKernel::operator()(
    Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<RowMatrix<double>> vnew,
    Eigen::Ref<Eigen::ArrayXd> acceptance_correction) {
  vnew = v;

  for (Index i = 0; i < v.rows(); i++) {
    hamiltonian_.FindConn(v.row(i), mel_, tochange_, newconfs_);

    auto w1 = static_cast<double>(tochange_.size());

    std::uniform_int_distribution<Index> distrs(0, tochange_.size() - 1);

    // picking a random state to transit to
    Index si = distrs(GetRandomEngine());

    // Inverse transition
    hamiltonian_.GetHilbert().UpdateConf(vnew.row(i), tochange_[si],
                                         newconfs_[si]);

    hamiltonian_.FindConn(vnew.row(i), mel_, tochange_, newconfs_);

    auto w2 = static_cast<double>(tochange_.size());

    acceptance_correction(i) = std::log(w1 / w2);
  }
}

}  // namespace netket
