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

#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#ifndef NETKET_FFNN_HPP
#define NETKET_FFNN_HPP

namespace netket {

using namespace std;
using namespace Eigen;

template <typename T> class FFNN : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;

  std::vector<std::unique_ptr<AbstractLayer<T>>>
      layers_; // Pointers to hidden layers

  std::vector<int> layersizes_;
  int depth_;
  int nlayer_;
  int npar_;

  int mynode_;

  const Hilbert &hilbert_;

public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit FFNN(const Hilbert &hilbert, const json &pars) : hilbert_(hilbert) {
    from_json(pars);
  }

  void from_json(const json &pars) override {
    if (FieldExists(pars["Machine"], "Layer Sizes")) {
      FieldArray(pars["Machine"], "Layer Sizes", layersizes_);
      depth_ = layersizes_.size();
    }

    Init(pars);
  }

  void Init(const json &pars) {

    layersizes_.push_back(1);
    nlayer_ = depth_;
    depth_ += 1;

    npar_ = 0;
    for (int i = 0; i < depth_ - 1; ++i) {
      npar_ += layersizes_[i + 1];
      npar_ += layersizes_[i + 1] * layersizes_[i];
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      cout << "# FFNN Initizialized with " << nlayer_ << " Layers: ";
      for (int i = 0; i < depth_ - 1; ++i) {
        cout << layersizes_[i] << " -> ";
      }
      cout << layersizes_[depth_ - 1];
      cout << endl;
    }
    // Initialise Layers
    for (int i = 0; i < nlayer_ - 1; ++i) {
      if (mynode_ == 0) {
        cout << "# Layer " << i + 1;
      }
      layers_.push_back(std::unique_ptr<FullyConnected<Lncosh, T>>(
          new FullyConnected<Lncosh, T>(layersizes_[i], layersizes_[i + 1])));
    }
    // identity output layer
    if (mynode_ == 0) {
      cout << "# Layer " << nlayer_;
    }
    layers_.push_back(std::unique_ptr<FullyConnected<Identity, T>>(
        new FullyConnected<Identity, T>(layersizes_[depth_ - 2], 1)));

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
  }

  int Nvisible() const override { return layersizes_[0]; }

  int Npar() const override { return npar_; }

  VectorType GetParameters() override {
    VectorType pars(npar_);
    int start_idx = 0;
    for (auto const &layer : layers_) {
      layer->GetParameters(pars, start_idx);
      start_idx += layer->Npar();
    }
    return pars;
  }

  void SetParameters(const VectorType &pars) override {
    int start_idx = 0;
    for (auto const &layer : layers_) {
      layer->SetParameters(pars, start_idx);
      start_idx += layer->Npar();
    }
  }

  void InitRandomPars(int seed, double sigma) override {
    for (auto const &layer : layers_) {
      layer->InitRandomPars(seed, sigma);
    }
  }

  void InitLookup(const VectorXd &v, LookupType &lt) override {
    layers_[0]->InitLookup(v, lt);
  }

  void UpdateLookup(const VectorXd &v, const vector<int> &tochange,
                    const vector<double> &newconf, LookupType &lt) override {
    layers_[0]->UpdateLookup(v, tochange, newconf, lt);
  }

  T LogVal(const VectorXd &v) override {
    // First layer
    layers_[0]->Forward(v);
    // The following layers
    for (int i = 1; i < depth_ - 1; i++) {
      layers_[i]->Forward(layers_[i - 1]->Output());
    }
    return (layers_.back()->Output())(0);
  }

  T LogVal(const VectorXd &v, const LookupType &lt) override {
    // First layer
    layers_[0]->Forward(v, lt);
    // The following layers
    for (int i = 1; i < depth_ - 1; i++) {
      layers_[i]->Forward(layers_[i - 1]->Output());
    }
    return (layers_.back()->Output())(0);
  }

  VectorType DerLog(const VectorXd &v) override {
    VectorType der(npar_);
    int start_idx = 0;

    VectorType last_dLda(1);
    last_dLda(0) = 1.0;
    // Forward pass
    LogVal(v);

    // Backpropagation
    if (nlayer_ > 1) {
      // Last Layer
      layers_[nlayer_ - 1]->Backprop(layers_[nlayer_ - 2]->Output(), last_dLda);
      // Middle Layers
      for (int i = nlayer_ - 2; i > 0; --i) {
        layers_[i]->Backprop(layers_[i - 1]->Output(),
                             layers_[i + 1]->Backprop_data());
      }
      // First Layer
      layers_[0]->Backprop(v, layers_[1]->Backprop_data());
    } else {
      // Only 1 layer
      layers_[0]->Backprop(v, last_dLda);
    }
    // Write Derivatives
    for (int i = 0; i < nlayer_; ++i) {
      layers_[i]->GetDerivative(der, start_idx);
      start_idx += layers_[i]->Npar();
    }
    return der;
  }

  VectorType LogValDiff(const VectorXd &v, const vector<vector<int>> &tochange,
                        const vector<vector<double>> &newconf) override {
    const int nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    LookupType theta;
    layers_[0]->InitLookup(v, theta);
    T current_val = LogVal(v, theta);

    for (int k = 0; k < nconn; ++k) {
      logvaldiffs(k) = 0;
      if (tochange[k].size() != 0) {
        LookupType theta_new;
        theta_new.AddVector(layersizes_[1]);
        theta_new.V(0) = theta.V(0);
        layers_[0]->UpdateLookup(v, tochange[k], newconf[k], theta_new);

        logvaldiffs(k) += LogVal(v, theta_new) - current_val;
      }
    }

    return logvaldiffs;
  }

  T LogValDiff(const VectorXd &v, const vector<int> &tochange,
               const vector<double> &newconf, const LookupType &lt) override {
    if (tochange.size() != 0) {
      T current_val = LogVal(v, lt);

      LookupType theta_new;
      theta_new.AddVector(layersizes_[1]);
      theta_new.V(0) = lt.V(0);

      layers_[0]->UpdateLookup(v, tochange, newconf, theta_new);

      return LogVal(v, theta_new) - current_val;
    } else {
      return 0.0;
    }
  }
  void to_json(json &j) const override {}
};

} // namespace netket

#endif
