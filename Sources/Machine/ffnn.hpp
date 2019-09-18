// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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

#ifndef NETKET_FFNN_HPP
#define NETKET_FFNN_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include "Layers/layer.hpp"
#include "Machine/abstract_machine.hpp"
#include "Utils/all_utils.hpp"

namespace netket {

class FFNN : public AbstractMachine {
  std::vector<AbstractLayer *> layers_;  // Pointers to hidden layers

  std::vector<int> layersizes_;
  int depth_;
  int nlayer_;
  int npar_;
  int nv_;
  std::vector<VectorType> din_;

  std::vector<std::vector<int>> changed_nodes_;
  std::vector<VectorType> new_output_;

  std::unique_ptr<SumOutput> sum_output_layer_;

  std::vector<VectorType> Vforward_;

 public:
  explicit FFNN(std::shared_ptr<const AbstractHilbert> hilbert,
                std::vector<AbstractLayer *> layers)
      : AbstractMachine(hilbert),
        layers_(std::move(layers)),
        nv_(hilbert->Size()) {
    Init();
  }

  void Init() {
    nlayer_ = layers_.size();

    std::string buffer = "";
    // Check that layer sizes are consistent
    layersizes_.push_back(nv_);
    for (int i = 0; i < nlayer_; ++i) {
      layersizes_.push_back(layers_[i]->Noutput());

      if (layersizes_[i] != layers_[i]->Ninput()) {
        throw InvalidInputError("input/output layer sizes do not match");
      }
    }

    // Check that final layer has only 1 unit otherwise add pooling layer
    if (layersizes_.back() != 1) {
      nlayer_ += 1;

      sum_output_layer_ = netket::make_unique<SumOutput>(layersizes_.back());
      layers_.push_back(sum_output_layer_.get());
      layersizes_.push_back(1);
    }
    depth_ = layersizes_.size();

    din_.resize(depth_);
    din_.back().resize(1);
    din_.back()(0) = 1.0;

    npar_ = 0;
    for (int i = 0; i < nlayer_; ++i) {
      npar_ += layers_[i]->Npar();
    }

    for (int i = 0; i < nlayer_; ++i) {
      Vforward_.push_back(VectorXcd(layersizes_[i + 1]));
    }

    changed_nodes_.resize(nlayer_);
    new_output_.resize(nlayer_);

    InfoMessage(buffer) << "# FFNN Initizialized with " << nlayer_
                        << " Layers: ";
    for (int i = 0; i < depth_ - 1; ++i) {
      InfoMessage(buffer) << layersizes_[i] << " -> ";
    }
    InfoMessage(buffer) << layersizes_[depth_ - 1];
    InfoMessage(buffer) << std::endl;
    for (int i = 0; i < nlayer_; ++i) {
      InfoMessage(buffer) << "# Layer " << i + 1 << " : " << layers_[i]->Name()
                          << std::endl;
    }
    InfoMessage(buffer) << "# Total Number of Parameters = " << npar_
                        << std::endl;
  }

  int Nvisible() const override { return layersizes_[0]; }

  int Npar() const override { return npar_; }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    int start_idx = 0;
    for (auto const layer : layers_) {
      int num_of_pars = layer->Npar();
      layer->GetParameters(pars.segment(start_idx, num_of_pars));
      start_idx += num_of_pars;
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int start_idx = 0;
    for (auto const layer : layers_) {
      int num_of_pars = layer->Npar();
      layer->SetParameters(pars.segment(start_idx, num_of_pars));
      start_idx += num_of_pars;
    }
  }

  void Forward(VisibleConstType v) {
    layers_[0]->Forward(v, Vforward_[0]);
    for (int i = 1; i < nlayer_; ++i) {
      layers_[i]->Forward(Vforward_[i - 1], Vforward_[i]);
    }
  }

  Complex LogValSingle(VisibleConstType v, const any & /*lookup*/) override {
    Forward(v);
    return Vforward_[nlayer_ - 1](0);
  }

  VectorType DerLogSingle(VisibleConstType v, const any & /*lookup*/) override {
    Forward(v);
    return DerLogSingleImpl(v);
  }

  VectorType DerLogSingleImpl(VisibleConstType v) {
    VectorType der(npar_);

    Index start_idx = npar_;
    Index num_of_pars;

    // Backpropagation
    if (nlayer_ > 1) {
      num_of_pars = layers_[nlayer_ - 1]->Npar();
      start_idx -= num_of_pars;
      // Last Layer
      layers_[nlayer_ - 1]->Backprop(
          Vforward_[nlayer_ - 2], Vforward_[nlayer_ - 1], din_.back(),
          din_[nlayer_ - 1], der.segment(start_idx, num_of_pars));
      // Middle Layers
      for (int i = nlayer_ - 2; i > 0; --i) {
        num_of_pars = layers_[i]->Npar();
        start_idx -= num_of_pars;
        layers_[i]->Backprop(Vforward_[i - 1], Vforward_[i], din_[i + 1],
                             din_[i], der.segment(start_idx, num_of_pars));
      }
      // First Layer
      layers_[0]->Backprop(v, Vforward_[0], din_[1], din_[0],
                           der.segment(0, layers_[0]->Npar()));
    } else {
      // Only 1 layer
      layers_[0]->Backprop(v, Vforward_[0], din_.back(), din_[0], der);
    }
    return der;
  }

  void Save(std::string const &filename) const override {
    json state;
    state["Name"] = "FFNN";
    state["Layers"] = {};
    for (int i = 0; i < nlayer_; ++i) {
      layers_[i]->to_json(state);
    }
    WriteJsonToFile(state, filename);
  }

  void Load(const std::string &filename) override {
    auto const pars = ReadJsonFromFile(filename);
    json layers_par;
    if (FieldExists(pars, "Layers")) {
      layers_par = pars["Layers"];
      nlayer_ = layers_par.size();
    } else {
      throw InvalidInputError(
          "Field (Layers) not defined for Machine (FFNN) in initfile");
    }

    for (int i = 0; i < nlayer_; ++i) {
      layers_[i]->from_json(layers_par[i]);
    }
  }

  bool IsHolomorphic() const noexcept override { return true; }
};  // namespace netket

}  // namespace netket

#endif
