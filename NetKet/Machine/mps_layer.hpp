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

#ifndef NETKET_MPSLAYER_HH
#define NETKET_MPSLAYER_HH

#include "abstract_layer.hpp"

namespace netket {
/** Recurrent neural network layer of matrix product state form, i.e.,
    using a tensor of rank 3 to compute the next hidden state.
 */
template <typename Activation, typename T>
class MpsLayer : public AbstractLayer<T> {

  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;
  static_assert(!MatrixType::IsRowMajor, "MatrixType must be column-major");

  Activation activation_;  // activation function class

  int nv_;      // number of sites (visible input units)
  int ls_;      // local size of Hilbert space
  int nh_;      // number of hidden units at each step
  int npar_;    // number of parameters in layer

  // weight parameters
  MatrixType W_;      // weight tensor W(ls_ x nh_ x nh_), stored as (ls_ * nh_) x nh_ matrix
  VectorType bias_;   // b(nh_ x 1)

 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  MpsLayer(const int nv, const int ls, const int nh)
      : activation_(), nv_(nv), ls_(ls), nh_(nh) {
    Init();
  }

  explicit MpsLayer(const Graph &graph, const json &pars)
      : activation_(), nv_(graph.Nsites()) {
    ls_ = FieldVal(pars, "LocalSize");
    nh_ = FieldVal(pars, "HiddenUnits");

    Init();
  }

  void Init() {
    W_.resize(ls_ * nh_, nh_);
    bias_.resize(nh_);

    npar_ = ls_ * nh_ * nh_ + nh_;

    std::string buffer = "";

    InfoMessage(buffer) << "RNN Layer of MPS form: " << Ninput() << " --> " << Noutput() << std::endl;
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

  int Noutput() const override { return nh_ * nv_; }

  void GetParameters(VectorType &pars, int start_idx) const override {
    int n = start_idx;

    for (int k = 0; k < nh_; ++k) {
      for (int j = 0; j < nh_; ++j) {
        for (int i = 0; i < ls_; ++i) {
          pars(n) = W_(i + ls_*j, k);
          ++n;
        }
      }
    }

    for (int i = 0; i < nh_; ++i) {
      pars(n) = bias_(i);
      ++n;
    }
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    int n = start_idx;

    for (int k = 0; k < nh_; ++k) {
      for (int j = 0; j < nh_; ++j) {
        for (int i = 0; i < ls_; ++i) {
          W_(i + ls_*j, k) = pars(n);
          ++n;
        }
      }
    }

    for (int i = 0; i < nh_; ++i) {
      bias_(i) = pars(n);
      ++n;
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
      // h_t = activation(kron(x_t, h_{t-1}) W + b)
      // form temporary Kronecker product of x_t and h_{t-1}
      const MatrixType xh = theta[0].segment(t*ls_, ls_) * h.transpose();
      activation_(W_.transpose() * Eigen::Map<const VectorType>(xh.data(), xh.size()) + bias_, h);

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

    MatrixType dW = MatrixType::Zero(ls_ * nh_, nh_);
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

      const VectorType h = this_layer_output.segment(t*nh_, nh_);

      const VectorType x = this_layer_theta[0].segment(t*ls_, ls_);

      const MatrixType xh = x * prev_h.transpose();

      // reconstruct argument of (nonlinear) activation function
      VectorType Z = W_.transpose() * Eigen::Map<const VectorType>(xh.data(), xh.size()) + bias_;
      VectorType dtanh(nh_);
      activation_.ApplyJacobian(Z, h, dh_accum + dout.segment(t*nh_, nh_), dtanh);

      // gradient with respect to input
      const MatrixType h_dtanh = prev_h * dtanh.transpose();  // outer product of two vectors
      // reshape W into a ls_ x (nh_ * nh_) matrix
      const auto W_reshape = Eigen::Map<const MatrixType>(W_.data(), ls_, nh_*nh_);
      din.segment(t*ls_, ls_) = W_reshape * Eigen::Map<const VectorType>(h_dtanh.data(), h_dtanh.size());

      // accumulated gradient of hidden state
      // reorder entries of W into a nh_ x (nh_ * ls_) matrix (cyclic permutation of dimensions)
      MatrixType W_reshape_T = W_reshape;
      W_reshape_T.transposeInPlace();
      const auto W_reorder = Eigen::Map<const MatrixType>(W_reshape_T.data(), nh_, nh_*ls_);
      const MatrixType dtanh_x = dtanh * x.transpose();  // outer product of two vectors
      dh_accum = W_reorder * Eigen::Map<const VectorType>(dtanh_x.data(), dtanh_x.size());

      dW += Eigen::Map<const VectorType>(xh.data(), xh.size()) * dtanh.transpose();
      db += dtanh;
    }

    // assign gradients of weight and bias parameters

    int n = start_idx;

    Eigen::Map<MatrixType> der_W{der.data() + n, ls_*nh_, nh_};
    der_W.noalias() = dW;
    n += ls_*nh_*nh_;

    Eigen::Map<VectorType> der_b{der.data() + n, nh_};
    der_b.noalias() = db;
    // avoid Codacy code review issue
    //n += nh_;
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "MpsLayer";
    layerpar["LocalSize"] = ls_;
    layerpar["HiddenUnits"] = nh_;
    layerpar["W"] = W_;
    layerpar["Bias"] = bias_;

    pars["Machine"]["Layers"].push_back(layerpar);
  }

  void from_json(const json &pars) override {
    if (FieldExists(pars, "W")) {
      W_ = pars["W"];
    } else {
      W_.setZero();
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
