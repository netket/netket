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

#ifndef NETKET_ABSTRACT_OPERATOR_HPP
#define NETKET_ABSTRACT_OPERATOR_HPP

#include <complex>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <nonstd/span.hpp>

#include "Hilbert/hilbert.hpp"

namespace netket {
/**
 * This struct represents a non-zero matrix-element H(v,v') of an operator for a
 * given visible state v.
 */
struct ConnectorRef {
  /// The matrix element H(v,v')
  Complex mel;
  /// The indices at which v needs to be changed to obtain v'
  nonstd::span<const int> tochange;
  /// The new values such that
  ///    v'(tochange[k]) = newconf[k]
  /// and v'(i) = v(i) for i ∉ tochange.
  nonstd::span<const double> newconf;
};

/**
      Abstract class for quantum Operators.
      This class prototypes the methods needed
      by a class satisfying the Operator concept.
      Users interested in implementing new quantum Operators should derive they
   own class from this class.
*/
class AbstractOperator {
 public:
  using VectorType = Eigen::VectorXd;
  using VectorRefType = Eigen::Ref<VectorType>;
  using VectorConstRefType = Eigen::Ref<const VectorType>;
  using MelType = std::vector<Complex>;
  using ConnectorsType = std::vector<std::vector<int>>;
  using NewconfsType = std::vector<std::vector<double>>;
  using ConnCallback = std::function<void(ConnectorRef)>;

  /**
  Member function finding the connected elements of the Operator.
  Starting from a given visible state v, it finds all other visible states v'
  such that the matrix element O(v,v') is different from zero.
  In general there will be several different connected visible units satisfying
  this condition, and they are denoted here v'(k), for k=0,1...N_connected.
  @param v a constant reference to the visible configuration.
  @param mel(k) is modified to contain matrix elements O(v,v'(k)).
  @param connector(k) for each k contains a list of sites that should be changed
  to obtain v'(k) starting from v.
  @param newconfs(k) is a vector containing the new values of the visible units
  on the affected sites, such that: v'(k,connectors(k,j))=newconfs(k,j). For the
  other sites v'(k)=v, i.e. they are equal to the starting visible
  configuration.
  */
  virtual void FindConn(VectorConstRefType v, MelType &mel,
                        ConnectorsType &connectors,
                        NewconfsType &newconfs) const = 0;

  virtual std::tuple<MelType, ConnectorsType, NewconfsType> GetConn(
      VectorConstRefType v) const;

  /**
   * Iterates over all states reachable from a given visible configuration v,
   * i.e., all states v' such that O(v,v') is non-zero.
   * @param v The visible configuration.
   * @param callback Function void callback(ConnectorRef conn) which will be
   * called once for each reachable configuration v'. The parameter conn
   * contains the value O(v,v') and the information to obtain v' from v. Note
   * that the members conn.positions and conn.values are spans that can only be
   * savely used inside the callback. They will become invalid once callback
   * returns.
   */
  virtual void ForEachConn(VectorConstRefType v, ConnCallback callback) const;

  /**
  Member function returning the hilbert space associated with this Hamiltonian.
  @return Hilbert space specifier for this Hamiltonian
  */
  const AbstractHilbert &GetHilbert() const { return *hilbert_; }

  std::shared_ptr<const AbstractHilbert> GetHilbertShared() const {
    return hilbert_;
  }

  /**
   * Returns the number of states in the underlying Hilbert space.
   *
   * @precondition `this->GetHilbert().IsIndexable()`
   */
  int Dimension() const { return GetHilbert().GetIndex().NStates(); }

  /**
   * Applies the operator to a quantum state.
   * @param state The entry state(i) corresponds to the coefficient of the basis
   * vector with quantum numbers given by StateFromNumber(i) of a HilbertIndex
   * for the original operator.
   * @precondition `this->GetHilbert().IsIndexable()`
   */
  virtual Eigen::VectorXcd Apply(const Eigen::VectorXcd &state) const {
    auto const &hilbert_index = GetHilbert().GetIndex();
    Eigen::VectorXcd result(hilbert_index.NStates());
    result.setZero();

    for (int i = 0; i < hilbert_index.NStates(); ++i) {
      // TODO: Make NumberToState return a reference so that we can avoid the
      // copy here.
      const auto v = hilbert_index.NumberToState(i);
      ForEachConn(v, [&](ConnectorRef conn) {
        const auto j = i + hilbert_index.DeltaStateToNumber(v, conn.tochange,
                                                            conn.newconf);
        result(i) += conn.mel * state(j);
      });
    }
    return result;
  }

  Eigen::MatrixXcd ToDense() const {
    const auto &hilbert_index = GetHilbert().GetIndex();
    Eigen::MatrixXcd matrix;
    matrix.resize(hilbert_index.NStates(), hilbert_index.NStates());
    matrix.setZero();
    ForEachMatrixElement([&matrix](const int i, const int j, const Complex x) {
      matrix(i, j) = x;
    });
    return matrix;
  }

  Eigen::SparseMatrix<Complex, Eigen::RowMajor> ToSparse() const {
    const auto &hilbert_index = GetHilbert().GetIndex();
    std::vector<Eigen::Triplet<Complex>> triplets;
    triplets.reserve(hilbert_index.NStates());

    ForEachMatrixElement(
        [&triplets](const int i, const int j, const Complex x) {
          triplets.emplace_back(i, j, x);
        });

    Eigen::SparseMatrix<Complex> matrix(hilbert_index.NStates(),
                                        hilbert_index.NStates());
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
  }

  virtual ~AbstractOperator() = default;

 protected:
  AbstractOperator(std::shared_ptr<const AbstractHilbert> hilbert)
      : hilbert_(std::move(hilbert)) {}

 private:
  template <class Function>
  void ForEachMatrixElement(Function &&function) const {
    const auto &hilbert_index = GetHilbert().GetIndex();
    for (int i = 0; i < hilbert_index.NStates(); ++i) {
      // TODO: Make NumberToState return a reference so that we can avoid the
      // copy here.
      const auto v = hilbert_index.NumberToState(i);
      ForEachConn(v, [&](ConnectorRef conn) {
        const auto j = i + hilbert_index.DeltaStateToNumber(v, conn.tochange,
                                                            conn.newconf);
        function(i, j, conn.mel);
      });
    }
  }

  std::shared_ptr<const AbstractHilbert> hilbert_;
};

template <class Operator>
Complex Mean(Operator &&apply, const Eigen::VectorXcd &state) {
  return state.adjoint() * std::forward<Operator>(apply)(state);
}

template <class Operator>
std::pair<Complex, double> MeanVariance(Operator &&apply,
                                        const Eigen::VectorXcd &state) {
  Eigen::VectorXcd state1 = apply(state);
  const Complex mean = state.adjoint() * state1;
  state1 -= mean * state;
  const double var = state1.norm();
  return {mean, var};
}

}  // namespace netket

#endif
