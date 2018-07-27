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

  /**
  Member function returning the number of inputs a layer takes.
  @return Number of Inputs into the Layer.
  */
  virtual int Ninput() const = 0;

  /**
  Member function returning the number of outputs from the layer.
  @return Number of Outputs from the Layer.
  */
  virtual int Noutput() const = 0;

  /**
  Member function returning the number of variational parameters.
  @return Number of variational parameters in the Layer.
  */
  virtual int Npar() const = 0;

  /**
  Member function writing the current set of parameters in the machine.
  @param pars is where the layer parameters are written into.
  @param start_idx is the index of the vector pars to start writing from.
  */
  virtual void GetParameters(VectorType &pars, int start_idx) const = 0;

  /**
  Member function setting the current set of parameters in the layer.
  @param pars is where the layer parameters are to be read from.
  @param start_idx is the index of the vector pars to start reading from.
  */
  virtual void SetParameters(const VectorType &pars, int start_idx) = 0;

  /**
  Member function providing a random initialization of the parameters.
  @param seed is the see of the random number generator.
  @param sigma is the variance of the gaussian.
  */
  virtual void InitRandomPars(int seed, double sigma) = 0;

  /**
  Member function to update the lookuptable which stores the output of each
  layer.
  @param v is a vector in the lookuptable storing the previous(old) input into
  the layer. This vector also corresponds to the output of the previous layer
  after the nonlinear transformation is applied. This vector is updated to the
  newconf at the end of the function.
  @param tochange is a vector containing the nodes of the input which has
  changed.
  @param newconf is a vector containing the new values at the changed nodes.
  @param theta is a vector in the lookuptable storing the output of the layer
  before the nonlinear function is applied.
  */
  virtual void UpdateLookup(VectorType &v, const std::vector<int> &tochange,
                            const VectorType &newconf, VectorType &theta) = 0;
  /**
  Member function to update the lookuptable which stores the output of each
  layer.
  @param v is a vector storing the previous(old) input into
  the layer. This vector does not belong in the lookuptable. It is the input
  configuration into the machine class.
  @param tochange is a vector containing the nodes of the input which has
  changed.
  @param newconf is a vector containing the new values at the changed nodes.
  @param theta is a vector in the lookuptable storing the output of the layer
  before the nonlinear function is applied.
  */
  virtual void UpdateLookup(const Eigen::VectorXd &v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            VectorType &theta) = 0;
  /**
  Member function to generate the nodes in the output of the current layer which
  have changed due to a change in the input. If the entire input is changed
  obviously all nodes in the output would also change, however if only a subset
  of input nodes are altered, depending on the type of layer, possibly only a
  subset of output nodes would change. Based on tochange and theta, the function
  would write the vector tochange1 and newconf1.
  @param theta is a vector in the lookuptable storing the output of the layer
  before the nonlinear function is applied.
  @param tochange is a vector containing the nodes of the input which has
  changed.
  @param newconf is a vector containing the new values at the changed nodes.
  @param tochange1 is a vector containing the nodes of the output which has
  changed.
  @param newconf1 is a vector containing the new values at the changed nodes.
  */
  virtual void NextConf(const VectorType &theta,
                        const std::vector<int> &tochange,
                        std::vector<int> &tochange1, VectorType &newconf1) = 0;

  /**
  Member function to update the vector in the lookuptable storing the output of
  the layer after the nonlinear function is applied.
  @param tochange1 is a vector containing the nodes of the output which has
  changed.
  @param newconf1 is a vector containing the new values at the changed nodes.
  @param v is a vector in the lookuptable storing the output of
  the layer after the nonlinear function is applied.
  */
  virtual void UpdateConf(const std::vector<int> &tochange1,
                          const VectorType &newconf1, VectorType &v) = 0;

  /**
  Member function to feedforward through the layer. Writes the output in to
  output, and the intermediate value after the linear transformation but before
  a nonliner transformation into theta
  @param prev_layer_output a constant reference to the output from previous
  layer.
  @param theta reference to the intermediate before the nonlinear transformation
  is applied.
  @param output reference to the output vector.
  */
  virtual void Forward(const VectorType &prev_layer_output, VectorType &theta,
                       VectorType &output) = 0;

  /**
  Member function to feedforward through the layer. Writes the output in to
  output, assuming the intermediate value theta has already been computed.
  @param theta reference to the intermediate before the nonlinear transformation
  is applied.
  @param output reference to the output vector.
  */
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
