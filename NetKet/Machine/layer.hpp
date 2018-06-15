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

#include "abstract_layer.hpp"
#include "activations.hpp"
#include "fullconn_layer.hpp"

#ifndef NETKET_LAYER_HPP
#define NETKET_LAYER_HPP

namespace netket {
template <class T>
class Layer : public AbstractLayer<T> {
  using Ptype = std::unique_ptr<AbstractLayer<T>>;

  Ptype m_;

 public:
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  explicit Layer(const json &pars) { Init(pars); }

  void Init(const json &pars) {
    if (pars["Name"] == "FullyConnected") {
      if (pars["Activation"] == "Lncosh") {
        m_ = Ptype(new FullyConnected<Lncosh, T>(pars));
      } else if (pars["Activation"] == "Identity") {
        m_ = Ptype(new FullyConnected<Identity, T>(pars));
      }
    }
  }

  int Npar() const override { return m_->Npar(); }

  int Ninput() const override { return m_->Ninput(); }

  int Noutput() const override { return m_->Noutput(); }

  void GetParameters(VectorType &pars, int start_idx) override {
    return m_->GetParameters(pars, start_idx);
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    return m_->SetParameters(pars, start_idx);
  }

  void InitRandomPars(int seed, double sigma) override {
    return m_->InitRandomPars(seed, sigma);
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    return m_->InitLookup(v, lt);
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    return m_->UpdateLookup(v, tochange, newconf, lt);
  }

  void Forward(const VectorType &prev_layer_data) override {
    return m_->Forward(prev_layer_data);
  }

  void Forward(const VectorType &prev_layer_data,
               const LookupType &lt) override {
    return m_->Forward(prev_layer_data, lt);
  }

  VectorType Output() const override { return m_->Output(); }

  void Backprop(const VectorType &prev_layer_data,
                const VectorType &next_layer_data) override {
    return m_->Backprop(prev_layer_data, next_layer_data);
  }

  const VectorType &Backprop_data() const override {
    return m_->Backprop_data();
  }

  void GetDerivative(VectorType &der, int start_idx) override {
    return m_->GetDerivative(der, start_idx);
  }
};
}  // namespace netket
#endif
