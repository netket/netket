// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "Machine/rbm_spin.hpp"

#include <pybind11/eigen.h>
#include <pybind11/eval.h>

#include "Utils/log_cosh.hpp"
#include "Utils/pybind_helpers.hpp"

namespace netket {

RbmSpin::RbmSpin(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
                 Index alpha, bool usea, bool useb)
    : AbstractMachine{std::move(hilbert)},
      W_{},
      a_{nonstd::nullopt},
      b_{nonstd::nullopt},
      theta_{} {
  const auto nvisible = GetHilbert().Size();
  assert(nvisible >= 0 && "AbstractHilbert::Size is broken");

  NETKET_CHECK(nhidden >= 0, InvalidInputError,
               "invalid number of hidden units: "
                   << nhidden << "; expected a non-negative number");

  NETKET_CHECK(alpha >= 0, InvalidInputError,
               "invalid density of hidden units: "
                   << alpha << "; expected a non-negative number");

  if (nhidden > 0 && alpha > 0) {
    NETKET_CHECK(nhidden == alpha * nvisible, InvalidInputError,
                 "number and density of hidden units are incompatible: "
                     << nhidden << " != " << alpha << " * " << nvisible);
  }
  nhidden = std::max(nhidden, alpha * nvisible);
  /* Initialization of the instance of RbmSpin class (defined in 
   * header) Variables are:
   * hilbert - the variable containing information about physical system
   * nhidden - number of neurons in hidden layer defined by larger of the values
   *           (alpha * number of neuron from visible layer) or nhidden
   * alpha - density defined as (number of neurons in hidden layer / number of
   *         neurons in visible layer)
   * usea - if true use biases in visible layer a*output
   * useb - if true use biases in hidden layer b*output
   */ 
        
  /* Defining variational parameters: weights and biases (a_ for 
   * visible layer, b_ for hidden layer)
   */
  W_.resize(nvisible, nhidden); 
  if (usea) {
    a_.emplace(nvisible);
  }
  if (useb) {
    b_.emplace(nhidden);
  }
  /* theta_ correspond to the "output" of a hidden layer 
   * theta = W_.transpose() * visible + b_. It is used as the input of cosh
   * functions in the expression for a wavefunction.
   */ 
  theta_.resize(1, nhidden);
}

/* Member function returning number of rows in theta_ (inherited from
 * AbstractMachine)
 */  
Index RbmSpin::BatchSize() const noexcept { return theta_.rows(); }

// Initialization function  
void RbmSpin::BatchSize(Index batch_size) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  if (batch_size != BatchSize()) {
    theta_.resize(batch_size, theta_.cols());
  }
}

// Function that returns parameters stored in a_, b_, W_  
Eigen::VectorXcd RbmSpin::GetParameters() {
  Eigen::VectorXcd parameters(Npar());
  Index i = 0;
  if (a_.has_value()) {
    parameters.segment(i, a_->size()) = *a_;
    i += a_->size();
  }
  if (b_.has_value()) {
    parameters.segment(i, b_->size()) = *b_;
    i += b_->size();
  }
  parameters.segment(i, W_.size()) =
      Eigen::Map<Eigen::VectorXcd>(W_.data(), W_.size());
  return parameters;
}

// Function that set the parameters "parameters" to the variables a_, b_, W_  
void RbmSpin::SetParameters(Eigen::Ref<const Eigen::VectorXcd> parameters) {
  CheckShape(__FUNCTION__, "parameters", parameters.size(), Npar());
  Index i = 0;
  if (a_.has_value()) {
    *a_ = parameters.segment(i, a_->size());
    i += a_->size();
  }
  if (b_.has_value()) {
    *b_ = parameters.segment(i, b_->size());
    i += b_->size();
  }
  Eigen::Map<Eigen::VectorXcd>(W_.data(), W_.size()) =
      parameters.segment(i, W_.size());
}

// Value of the logarithm of the wave-function
// using pre-computed look-up matrix for efficiency  
void RbmSpin::LogVal(Eigen::Ref<const RowMatrix<double>> x,
                     Eigen::Ref<Eigen::VectorXcd> out, const any & /*lt*/) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), x.rows());
  BatchSize(x.rows());
  if (a_.has_value()) {
    out.noalias() = x * (*a_);
  } else {
    out.setZero();
  }
  theta_.noalias() = x * W_;
  ApplyBiasAndActivation(out);
}


/* Function that calculates the derivatives of logarithms of a 
 * wavefunction. It uses a look-up matrix.
 */ 
void RbmSpin::DerLog(Eigen::Ref<const RowMatrix<double>> x,
                     Eigen::Ref<RowMatrix<Complex>> out, const any & /*lt*/) {
  /* Eigen::RowMatrix<Complex, Eigen::Dynamic, 1> is equivalent to VectorType.
   * der is a vector containing all the derivatives of variational parameters. 
   * der.head correspond to the beginning of the vector, der.segment to the middle
   * and der.tail to the end of the vector
   */
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()}, {x.rows(), Npar()});
  BatchSize(x.rows());

  auto i = Index{0};
  /* derivative of log(wavefunction) with respect to a_ biases is given 
   * by values of visible layers
   */
  if (a_.has_value()) {
    out.block(0, i, BatchSize(), a_->size()) = x;
    i += a_->size();
  }

  Eigen::Map<RowMatrix<Complex>>{theta_.data(), theta_.rows(), theta_.cols()}
      .noalias() = x * W_;
  /* derivative of log(wavefunction) with respect to b_ biases is given
   * by theta_ * tanh
   */ 
  if (b_.has_value()) {
    theta_.array() = (theta_ + b_->transpose().colwise().replicate(BatchSize()))
                         .array()
                         .tanh();
    out.block(0, i, BatchSize(), b_->size()) = theta_;
    i += b_->size();
  } else {
    theta_.array() = theta_.array().tanh();
  }

  // TODO: Rewrite this using tensors
#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i), W_.rows(), W_.cols()}.noalias() =
        x.row(j).transpose() * theta_.row(j);
  }
}

// Loading the bias and activation parameters  
void RbmSpin::ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const {
  if (b_.has_value()) {
#pragma omp parallel for schedule(static)
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      out(j) += SumLogCoshBias(theta_.row(j), (*b_));  // total;
    }
  } else {
#pragma omp parallel for schedule(static)
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      out(j) += SumLogCosh(theta_.row(j));
    }
  }
}

// Saving the bias and activation parameters in a dictionary  
PyObject *RbmSpin::StateDict() {
  return ToStateDict(std::make_tuple(std::make_pair("a", std::ref(a_)),
                                     std::make_pair("b", std::ref(b_)),
                                     std::make_pair("w", std::ref(W_))));
}

}  // namespace netket
