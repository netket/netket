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

#ifndef NETKET_DENSE_HAMILTONIAN_OPERATOR_HPP
#define NETKET_DENSE_HAMILTONIAN_OPERATOR_HPP

#include <Eigen/Dense>

#include "Hilbert/hilbert_index.hpp"
#include "Operator/abstract_operator.hpp"
#include "abstract_matrix_wrapper.hpp"

namespace netket {

/**
 * This class stores the matrix elements of a given Operator
 * as an Eigen dense matrix.
 */
template <class State = Eigen::VectorXcd>
class DenseMatrixWrapper : public AbstractMatrixWrapper<State> {
 public:
  using Matrix = Eigen::MatrixXcd;

 private:
  Matrix matrix_;
  int dim_;

 public:
  explicit DenseMatrixWrapper(const AbstractOperator& the_operator) {
    InitializeMatrix(the_operator);
  }

  State Apply(const State& state) const override { return matrix_ * state; }

  Complex Mean(const State& state) const override {
    return state.adjoint() * matrix_ * state;
  }

  std::array<Complex, 2> MeanVariance(
      const State& state) const override {
    auto state1 = matrix_ * state;
    auto state2 = matrix_ * state1;

    const Complex mean = state.adjoint() * state1;
    const Complex var = state.adjoint() * state2;

    return {{mean, var - std::pow(mean, 2)}};
  }

  int Dimension() const override { return dim_; }

  /**
   * @return An Eigen::MatrixXcd containing the matrix elements of the wrapped
   * operator.
   */
  const Matrix& GetMatrix() const { return matrix_; }

  /**
   * Computes the eigendecomposition of the given matrix.
   * @param options The options are passed directly to the constructor of
   * SelfAdjointEigenSolver.
   * @return An instance of Eigen::SelfAdjointEigenSolver initialized with the
   * wrapped operator and options.
   */
  Eigen::SelfAdjointEigenSolver<Matrix> ComputeEigendecomposition(
      int options = Eigen::ComputeEigenvectors) const {
    return Eigen::SelfAdjointEigenSolver<Matrix>(matrix_, options);
  }

 private:
  void InitializeMatrix(const AbstractOperator& the_operator) {
    const auto& hilbert_index = the_operator.GetHilbert().GetIndex();
    dim_ = hilbert_index.NStates();

    matrix_.resize(dim_, dim_);
    matrix_.setZero();

    for (int i = 0; i < dim_; ++i) {
      const auto v = hilbert_index.NumberToState(i);
      the_operator.ForEachConn(v, [&](ConnectorRef conn) {
        const auto j = i + hilbert_index.DeltaStateToNumber(v, conn.tochange,
                                                            conn.newconf);
        matrix_(i, j) += conn.mel;
      });
    }
  }
};

}  // namespace netket

#endif  // NETKET_DENSE_HAMILTONIAN_OPERATOR_HPP
