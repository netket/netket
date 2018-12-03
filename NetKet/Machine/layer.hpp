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

#ifndef NETKET_LAYER_HPP
#define NETKET_LAYER_HPP

#include "abstract_layer.hpp"
#include "activation_layer.hpp"
#include "activations.hpp"
#include "fullconn_layer.hpp"
#include "hypercube_conv_layer.hpp"
#include "mpark/variant.hpp"
#include "sum_output.hpp"

namespace netket {
template <typename T>
class Layer : public AbstractLayer<T> {
 public:
  using VariantType =
      mpark::variant<FullyConnected<T>, ConvolutionalHypercube<T>, SumOutput<T>,
                     Activation<T, Lncosh>, Activation<T, Tanh>,
                     Activation<T, Relu>>;
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;
  using VectorRefType = typename AbstractLayer<T>::VectorRefType;
  using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

 private:
  VariantType obj_;

 public:
  Layer(VariantType obj) : obj_(obj) {}

  std::string Name() const override {
    return mpark::visit([](auto &&obj) { return obj.Name(); }, obj_);
  }

  int Ninput() const override {
    return mpark::visit([](auto &&obj) { return obj.Ninput(); }, obj_);
  }

  int Noutput() const override {
    return mpark::visit([](auto &&obj) { return obj.Noutput(); }, obj_);
  }

  int Npar() const override {
    return mpark::visit([](auto &&obj) { return obj.Npar(); }, obj_);
  }

  void GetParameters(VectorRefType pars) const override {
    mpark::visit([pars](auto &&obj) { obj.GetParameters(pars); }, obj_);
  }

  void SetParameters(VectorConstRefType pars) override {
    mpark::visit([pars](auto &&obj) { obj.SetParameters(pars); }, obj_);
  }

  void InitRandomPars(int seed, double sigma) override {
    mpark::visit([=](auto &&obj) { obj.InitRandomPars(seed, sigma); }, obj_);
  }

  void UpdateLookup(const VectorType &input,
                    const std::vector<int> &input_changes,
                    const VectorType &new_input, const VectorType &output,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    mpark::visit(
        [&](auto &&obj) {
          obj.UpdateLookup(input, input_changes, new_input, output,
                           output_changes, new_output);
        },
        obj_);
  }

  void Forward(const VectorType &input, VectorType &output) override {
    mpark::visit([&](auto &&obj) { obj.Forward(input, output); }, obj_);
  }

  void Backprop(const VectorType &prev_layer_output,
                const VectorType &this_layer_output, const VectorType &dout,
                VectorType &din, VectorRefType der) override {
    mpark::visit(
        [&, der](auto &&obj) {
          obj.Backprop(prev_layer_output, this_layer_output, dout, din, der);
        },
        obj_);
  }

  void to_json(json &j) const override {
    mpark::visit([&](auto &&obj) { obj.to_json(j); }, obj_);
  }

  void from_json(const json &j) override {
    mpark::visit([&](auto &&obj) { obj.from_json(j); }, obj_);
  }
};
}  // namespace netket
#endif
