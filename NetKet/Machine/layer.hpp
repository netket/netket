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
#include "conv_layer.hpp"
#include "fullconn_layer.hpp"
#include "sum_output.hpp"

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

  explicit Layer(const Graph &graph, const json &pars) { Init(graph, pars); }

  void Init(const Graph &graph, const json &pars) {
    CheckInput(pars);
    if (pars["Name"] == "FullyConnected") {
      if (pars["Activation"] == "Lncosh") {
        m_ = Ptype(new FullyConnected<Lncosh, T>(pars));
      } else if (pars["Activation"] == "Identity") {
        m_ = Ptype(new FullyConnected<Identity, T>(pars));
      }
    } else if (pars["Name"] == "Convolutional") {
      if (pars["Activation"] == "Lncosh") {
        m_ = Ptype(new Convolutional<Lncosh, T>(graph, pars));
      } else if (pars["Activation"] == "Identity") {
        m_ = Ptype(new Convolutional<Identity, T>(graph, pars));
      } else if (pars["Activation"] == "Tanh") {
        m_ = Ptype(new Convolutional<Tanh, T>(graph, pars));
      }
    } else if (pars["Name"] == "Sum") {
      m_ = Ptype(new SumOutput<T>(pars));
    }
  }

  void CheckInput(const json &pars) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    const std::string name = FieldVal(pars, "Name");

    std::set<std::string> layers = {"FullyConnected", "Convolutional",
                                    "Symmetric", "Sum"};

    if (layers.count(name) == 0) {
      std::stringstream s;
      s << "Unknown Machine: " << name;
      throw InvalidInputError(s.str());
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
                const VectorType &next_layer_data, VectorType &der,
                int start_idx) override {
    return m_->Backprop(prev_layer_data, next_layer_data, der, start_idx);
  }

  const VectorType &BackpropData() const override { return m_->BackpropData(); }

  void to_json(json &j) const override { m_->to_json(j); }

  void from_json(const json &j) override { m_->from_json(j); }
};  // namespace netket
}  // namespace netket
#endif
