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

#ifndef NETKET_ABSTRACT_MATRIX_WRAPPER_HPP
#define NETKET_ABSTRACT_MATRIX_WRAPPER_HPP

#include "Operator/abstract_operator.hpp"

namespace netket {

/**
 * This class wraps an AbstractOperator
 * and provides a method to apply it to a pure state.
 * @tparam WfType The type of a vector of (complex) coefficients representing
 * the wavefunction. Should be Eigen::VectorXcd or a compatible type.
 */
template <class Operator, class WfType = Eigen::VectorXcd>
class AbstractMatrixWrapper {
 public:
  /**
   * Applies the wrapped hamiltonian to a quantum state.
   * @param state The entry state(i) corresponds to the coefficient of the basis
   * vector with quantum numbers given by StateFromNumber(i) of a HilbertIndex
   * for the original Hamiltonian. The state should satisfy state.size() ==
   * GetDimension().
   */
  virtual WfType Apply(const WfType& state) const = 0;

  virtual std::complex<double> Mean(const WfType& state) const {
    return state.adjoint() * Apply(state);
  }

  virtual std::array<std::complex<double>, 2> MeanVariance(
      const WfType& state) const {
    auto state1 = Apply(state);
    auto state2 = Apply(state1);

    const std::complex<double> mean = state.adjoint() * state1;
    const std::complex<double> var = state.adjoint() * state2;

    return {{mean, var - std::pow(mean, 2)}};
  }

  /**
   * Returns the Hilbert space dimension corresponding to the Hamiltonian.
   */
  virtual int GetDimension() const = 0;

  virtual ~AbstractMatrixWrapper() = default;
};

}  // namespace netket

#endif  // NETKET_ABSTRACT_MATRIX_WRAPPER_HPP
