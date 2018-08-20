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
//
// by C. B. Mendl, August 2018

#ifndef NETKET_RECURRENTLAYER_HH
#define NETKET_RECURRENTLAYER_HH

#include "abstract_layer.hpp"

namespace netket {
/** Vanilla recurrent neural network layer.
    Assumes sequential ordering of input units;
    output consists of hidden states at all time points.
 */
template <typename Activation, typename T>
class Recurrent : public AbstractLayer<T> {

  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;

  Activation activation_;  // activation function class

  int nv_;      // number of sites (visible input units)
  int ls_;      // local size of Hilbert space
  int nh_;      // number of hidden units at each step
  int npar_;    // number of parameters in layer

  // weight parameters
  MatrixType W_;      // W(nh_ x nh_)
  MatrixType U_;      // U(nh_ x ls_)
  VectorType bias_;   // b(nh_ x 1)

 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  Recurrent(const int nv, const int ls, const int nh)
      : activation_(), nv_(nv), ls_(ls), nh_(nh) {
    Init();
  }

  explicit Recurrent(const Graph &graph, const json &pars)
      : activation_(), nv_(graph.Nsites()) {
    ls_ = FieldVal(pars, "LocalSize");
    nh_ = FieldVal(pars, "HiddenUnits");

    Init();
  }

  void Init() {
    W_.resize(nh_, nh_);
    U_.resize(nh_, ls_);
    bias_.resize(nh_);

    npar_ = nh_ * nh_ + nh_ * ls_ + nh_;

    std::string buffer = "";

    InfoMessage(buffer) << "Recurrent Layer: " << Ninput() << " --> " << Noutput() << std::endl;
    InfoMessage(buffer) << "# # Local size   = " << ls_ << std::endl;
    InfoMessage(buffer) << "# # Hidden units = " << nh_ << std::endl;
  }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    // scale by number of hidden units to avoid blow-up
    netket::RandomGaussian(par, seed, sigma / nh_);

    SetParameters(par, 0);
  }

  int Npar() const override { return npar_; }

  int Ninput() const override { return ls_ * nv_; }

  int Noutput() const override { return nv_ * nh_; }

  void GetParameters(VectorType &pars, int start_idx) const override {
    int k = start_idx;

    for (int j = 0; j < nh_; ++j) {
      for (int i = 0; i < nh_; ++i) {
        pars(k) = W_(i, j);
        ++k;
      }
    }

    for (int j = 0; j < ls_; ++j) {
      for (int i = 0; i < nh_; ++i) {
        pars(k) = U_(i, j);
        ++k;
      }
    }

    for (int i = 0; i < nh_; ++i) {
      pars(k) = bias_(i);
      ++k;
    }
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    int k = start_idx;

    for (int j = 0; j < nh_; ++j) {
      for (int i = 0; i < nh_; ++i) {
        W_(i, j) = pars(k);
        ++k;
      }
    }

    for (int j = 0; j < ls_; ++j) {
      for (int i = 0; i < nh_; ++i) {
        U_(i, j) = pars(k);
        ++k;
      }
    }

    for (int i = 0; i < nh_; ++i) {
      bias_(i) = pars(k);
      ++k;
    }
  }

  void InitLookup(const VectorType &v, LookupType &lt,
                  VectorType &output) override {
    lt.resize(1);
    lt[0].resize(nv_ * ls_);
    Forward(v, lt, output);
  }

  void UpdateLookup(const VectorType &/*input*/,
                    const std::vector<int> &input_changes,
                    const VectorType &new_input, LookupType &theta,
                    const VectorType & /*output*/,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    const int num_of_changes = input_changes.size();
    if (num_of_changes > 0) {
      output_changes.resize(Noutput());
      new_output.resize(Noutput());
      Forward(new_input, theta, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  void UpdateLookup(const Eigen::VectorXd &input,
                    const std::vector<int> &tochange,
                    const std::vector<double> &newconf, LookupType &theta,
                    const VectorType & /*output*/,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    // reconstruct new input
    VectorType new_input(input);
    const int num_of_changes = tochange.size();
    for (int s = 0; s < num_of_changes; ++s) {
      const int sf = tochange[s];
      new_input(sf) = newconf[s];
    }
    if (num_of_changes > 0) {
      output_changes.resize(Noutput());
      new_output.resize(Noutput());
      Forward(new_input, theta, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  // Feedforward
  void Forward(const VectorType &prev_layer_output, LookupType &theta,
               VectorType &output) override {
    // simply copy input since recurrent layer is inherently nonlinear
    theta[0].noalias() = prev_layer_output;

    Forward(theta, output);
  }

  // Feedforward using lookup
  void Forward(const LookupType &theta, VectorType &output) override {
    VectorType h = VectorType::Zero(nh_);

    for (int t = 0; t < nv_; ++t) {
      // h_t = activation(W h_{t-1} + U x_t + b)
      activation_(W_ * h + U_ * theta[0].segment(t*ls_, ls_) + bias_, h);

      // store current hidden state in output
      for (int k = 0; k < nh_; ++k) {
        output(nh_ * t + k) = h(k);
      }
    }
  }

  void Backprop(const VectorType &/*prev_layer_output*/,
                const VectorType &this_layer_output,
                const LookupType &this_layer_theta, const VectorType &dout,
                VectorType &din, VectorType &der, int start_idx) override {

    MatrixType dW = MatrixType::Zero(nh_, nh_);
    MatrixType dU = MatrixType::Zero(nh_, ls_);
    VectorType db = VectorType::Zero(nh_);
    VectorType dh_accum = VectorType::Zero(nh_);

    din.resize(nv_ * ls_);

    for (int t = nv_ - 1; t >= 0; --t) {

      VectorType prev_h;
      if (t > 0) {
        prev_h = this_layer_output.segment((t - 1)*nh_, nh_);
      }
      else {
        prev_h = VectorType::Zero(nh_);
      }

      const VectorType h_t = this_layer_output.segment(t*nh_, nh_);

      // reconstruct argument of (nonlinear) activation function
      auto Z = W_ * prev_h + U_ * this_layer_theta[0].segment(t*ls_, ls_) + bias_;
      VectorType dtanh(nh_);
      activation_.ApplyJacobian(Z, h_t, dh_accum + dout.segment(t*nh_, nh_), dtanh);
      din.segment(t*ls_, ls_) = dtanh.transpose() * U_;
      dh_accum = dtanh.transpose() * W_;
      dW += dtanh * prev_h.transpose();
      dU += dtanh * this_layer_theta[0].segment(t*ls_, ls_).transpose();  // outer product of two vectors
      db += dtanh;
    }

    // assign gradients of weight and bias parameters

    int k = start_idx;

    Eigen::Map<MatrixType> der_W{der.data() + k, nh_, nh_};
    der_W.noalias() = dW;
    k += nh_*nh_;

    Eigen::Map<MatrixType> der_U{der.data() + k, nh_, ls_};
    der_U.noalias() = dU;
    k += nh_*ls_;

    Eigen::Map<VectorType> der_b{der.data() + k, nh_};
    der_b.noalias() = db;
    // avoid Codacy code review issue
    //k += nh_;
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "Recurrent";
    layerpar["LocalSize"] = ls_;
    layerpar["HiddenUnits"] = nh_;
    layerpar["W"] = W_;
    layerpar["U"] = U_;
    layerpar["Bias"] = bias_;

    pars["Machine"]["Layers"].push_back(layerpar);
  }

  void from_json(const json &pars) override {
    if (FieldExists(pars, "W")) {
      W_ = pars["W"];
    } else {
      W_.setZero();
    }
    if (FieldExists(pars, "U")) {
      U_ = pars["U"];
    } else {
      U_.setZero();
    }
    if (FieldExists(pars, "Bias")) {
      bias_ = pars["Bias"];
    } else {
      bias_.setZero();
    }
  }
};
}  // namespace netket

#endif
