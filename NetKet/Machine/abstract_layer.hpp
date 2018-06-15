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

#ifndef NETKET_ABSTRACTLAYER_HH
#define NETKET_ABSTRACTLAYER_HH

#include <Eigen/Dense>
#include <Lookup/lookup.hpp>
#include <complex>
#include <fstream>
#include <netket.hpp>
#include <random>
#include <vector>

namespace netket {
/**
  Abstract class for Neural Network layer.
*/
template <typename T>
class AbstractLayer {
 public:
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using StateType = T;
  using LookupType = Lookup<T>;

  virtual int Ninput() const = 0;

  virtual int Noutput() const = 0;

  virtual int Npar() const = 0;

  virtual void GetParameters(VectorType &pars, int start_idx) = 0;

  virtual void SetParameters(const VectorType &pars, int start_idx) = 0;

  virtual void InitRandomPars(int seed, double sigma) = 0;

  virtual void InitLookup(const Eigen::VectorXd &v, LookupType &lt) = 0;

  virtual void UpdateLookup(const Eigen::VectorXd &v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            LookupType &lt) = 0;

  /**
  Member function to feedforward through the layer.
  @param prev_layer_data a constant reference to the output from previous layer.
  */
  virtual void Forward(const VectorType &prev_layer_data) = 0;

  virtual void Forward(const VectorType &prev_layer_data,
                       const LookupType &lt) = 0;
  /**
  Member function to return output of forward propagation.
  */
  virtual VectorType Output() const = 0;

  /**
  Member function to perform backpropagation to compute derivates.
  @param prev_layer_data a constant reference to the output from previous layer.
  @param next_layer_data a constant reference to the derivative dL/dA where A is
  the activations of the current layer and L is the the final output of the
  Machine: L = log(psi(v))
  */
  virtual void Backprop(const VectorType &prev_layer_data,
                        const VectorType &next_layer_data) = 0;
  /**
  Member function to return dL/d(in), where (in) is the input to the current
  layer, and L = log(psi(v))
  */
  virtual const VectorType &BackpropData() const = 0;

  /**
  Member function to write derivatives into der.
  @der reference to the vector containing derivatives of the whole machine.
  @start_idx index to indicate where to start writing derivatives in der.
  */
  virtual void GetDerivative(VectorType &der, int start_idx) = 0;

  /**
  destructor
  */
  virtual ~AbstractLayer() {}
};
}  // namespace netket

#endif
