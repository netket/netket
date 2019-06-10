// Copyright 2018 Damian Hofmann - All Rights Reserved.
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

#ifndef NETKET_DIRECT_HAMILTONIAN_OPERATOR_HPP
#define NETKET_DIRECT_HAMILTONIAN_OPERATOR_HPP

#include <complex>
#include <vector>

#include <Eigen/Dense>

#include "Hilbert/hilbert_index.hpp"
#include "Operator/abstract_operator.hpp"
#include "abstract_matrix_wrapper.hpp"

namespace netket {

/**
 * This class wraps a given Operator. The matrix elements are not stored
 * separately but are computed from Operator::FindConn every time Apply is
 * called.
 */
template <class State = Eigen::VectorXcd>
class DirectMatrixWrapper : public AbstractMatrixWrapper<State> {
  const AbstractOperator& operator_;
  const HilbertIndex& hilbert_index_;
  int dim_;

 public:
  explicit DirectMatrixWrapper(const AbstractOperator& the_operator)
      : operator_(the_operator),
        hilbert_index_(the_operator.GetHilbert().GetIndex()),
        dim_(hilbert_index_.NStates()) {}

  State Apply(const State& state) const override {
    State result(dim_);
    result.setZero();

    for (int i = 0; i < dim_; ++i) {
      const auto v = hilbert_index_.NumberToState(i);
      operator_.ForEachConn(v, [&](ConnectorRef conn) {
        const auto j = i + hilbert_index_.DeltaStateToNumber(v, conn.tochange,
                                                             conn.newconf);
        result(i) += conn.mel * state(j);
      });
    }
    return result;
  }

  int Dimension() const override { return dim_; }
};

}  // namespace netket

#endif  // NETKET_DIRECT_HAMILTONIAN_OPERATOR_HPP
