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
#include <Utils/lookup.hpp>
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
  using LookupType = std::vector<VectorType>;
  using VectorRefType = Eigen::Ref<VectorType>;
  using VectorConstRefType = Eigen::Ref<const VectorType>;

  /**
  Member function returning the name of the layer.
  @return Name of Layer.
  */
  virtual std::string Name() const = 0;
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
  virtual void GetParameters(VectorRefType pars, int start_idx) const = 0;

  /**
  Member function setting the current set of parameters in the layer.
  @param pars is where the layer parameters are to be read from.
  @param start_idx is the index of the vector pars to start reading from.
  */
  virtual void SetParameters(VectorConstRefType pars, int start_idx) = 0;

  /**
  Member function providing a random initialization of the parameters.
  @param seed is the see of the random number generator.
  @param sigma is the variance of the gaussian.
  */
  virtual void InitRandomPars(int seed, double sigma) = 0;
  /**
  Member function to initialise the lookup for the later. "Summary: This
  function uses the input vector v to compute the lookup lt and the output. This
  function also resizes the lookup elements to the correct size."
  @param v is the input vector.
  @param theta is a vector in the lookuptable storing the output of the
  layer before the nonlinear function is applied.
  @param output is a vector in the lookuptable storing the previous output of
  the current layer.
  */
  virtual void InitLookup(const VectorType &v, LookupType &lt,
                          VectorType &output) = 0;

  /**
  Member function to update the lookuptable which stores the theta of
  each layer. "Summary: This function uses input, input_changes and new_input
  to generate theta which in turn gives output on application of the nonlinear
  transformation. output_changes and new_output keeps track of how the
  output of the layer has changed due to changes in the input."
  @param input is a vector in the lookuptable storing the output from the
  previous layer. This would be the old input to the current layer.
  @param input_changes is a std::vector containing the nodes of the input
  which has changed.
  @param new_input is a vector containing the new values at the changed
  input nodes.
  @param theta is a vector in the lookuptable storing the output of the
  layer before the nonlinear function is applied.
  @param output is a vector in the lookuptable storing the previous output of
  the current layer.
  @param output_changes is a vector containing the nodes of the output which has
  changed.
  @param new_output is a vector storing the new output of the layer.
  */
  virtual void UpdateLookup(const VectorType &input,
                            const std::vector<int> &input_changes,
                            const VectorType &new_input, LookupType &theta,
                            const VectorType &output,
                            std::vector<int> &output_changes,
                            VectorType &new_output) = 0;
  /**
  Member function to update the lookuptable which stores the theta of
  each layer. "Summary: This function uses input, tochange and newconf
  to generate theta which in turn gives output on application of the nonlinear
  transformation. output_changes and new_output keeps track of how the
  output of the layer has changed due to changes in the input."
  @param input is a vector in the lookuptable storing the output from the
  previous layer. This would be the old input to the current layer.
  @param tochange is a std::vector containing the nodes of the input
  which has changed.
  @param newconf is a vector containing the new values at the changed
  input nodes.
  @param theta is a vector in the lookuptable storing the output of the
  layer before the nonlinear function is applied.
  @param output is a vector in the lookuptable storing the previous output of
  the current layer.
  @param output_changes is a vector containing the nodes of the output which has
  changed.
  @param new_output is a vector storing the new output of the layer.
  */
  virtual void UpdateLookup(const Eigen::VectorXd &input,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            LookupType &theta, const VectorType &output,
                            std::vector<int> &output_changes,
                            VectorType &new_output) = 0;

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
  virtual void Forward(const VectorType &prev_layer_output, LookupType &theta,
                       VectorType &output) = 0;

  /**
  Member function to perform backpropagation to compute derivates.
  @param prev_layer_output a constant reference to the output from previous
  layer.
  @param this_layer_output a constant reference to the output from the current
  layer.
  @param next_layer_data a constant reference to the derivative dL/dA where A is
  the activations of the current layer and L is the the final output of the
  Machine: L = log(psi(v))
  @param din a constant reference to the derivative of the input from the
  current layer.
  @param der a constant reference to the derivatives wrt to the parameters in
  the machine.
  */
  virtual void Backprop(const VectorType &prev_layer_output,
                        const VectorType &this_layer_output,
                        const VectorType &dout, VectorType &din,
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
