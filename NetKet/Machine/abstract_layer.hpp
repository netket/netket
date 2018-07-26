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

  virtual void GetParameters(VectorType &pars, int start_idx) const = 0;

  virtual void SetParameters(const VectorType &pars, int start_idx) = 0;

  virtual void InitRandomPars(int seed, double sigma) = 0;

  virtual void UpdateLookup(const VectorType &v,
                            const std::vector<int> &tochange,
                            const VectorType &newconf, VectorType &theta) = 0;

  virtual void UpdateLookup(const Eigen::VectorXd &v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            VectorType &theta) = 0;

  virtual void NextConf(const VectorType &theta,
                        const std::vector<int> &tochange,
                        std::vector<int> &tochange1, VectorType &newconf1) = 0;

  virtual void UpdateConf(const std::vector<int> &tochange,
                          const VectorType &newconf, VectorType &v) = 0;

  /**
  Member function to feedforward through the layer.
  @param prev_layer_output a constant reference to the output from previous
  layer.
  */
  virtual void Forward(const VectorType &prev_layer_output, VectorType &theta,
                       VectorType &output) = 0;

  virtual void Forward(const VectorType &theta, VectorType &output) = 0;

  /**
  Member function to perform backpropagation to compute derivates.
  @param prev_layer_output a constant reference to the output from previous
  layer.
  @param next_layer_data a constant reference to the derivative dL/dA where A is
  the activations of the current layer and L is the the final output of the
  Machine: L = log(psi(v))
  */
  virtual void Backprop(const VectorType &prev_layer_output,
                        const VectorType &this_layer_output,
                        const VectorType &this_layer_theta,
                        const VectorType &next_layer_data, VectorType &din,
                        VectorType &der, int start_idx) = 0;

  virtual void to_json(json &j) const = 0;

  virtual void from_json(const json &j) = 0;

  /**
  destructor
  */
  virtual ~AbstractLayer() {}
};
}  // namespace netket

#endif
