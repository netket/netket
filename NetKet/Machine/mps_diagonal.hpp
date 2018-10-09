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
// by S. Efthymiou, October 2018

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include "abstract_mps.hpp"

#ifndef NETKET_MPS_DIAGONAL_HPP
#define NETKET_MPS_DIAGONAL_HPP

namespace netket {

template <typename T>
class MPSDiagonal : public AbstractMPS<T> {
  using VectorType = typename AbstractMPS<T>::VectorType;
  using MatrixType = typename AbstractMPS<T>::MatrixType;

  // Number of sites
  int N_;
  // Physical dimension
  int d_;
  // Bond dimension
  int D_;
  // Number of variational parameters
  int npar_;
  // Period of translational symmetry (has to be a divisor of N)
  int symperiod_;

  // MPS Matrices (stored as [symperiod, d, D, D]
  std::vector<std::vector<VectorType>> W_;

  // Map from Hilbert states to MPS indices
  std::map<double, int> confindex_;

  const Hilbert &hilbert_;

 public:
  using StateType = T;
  using LookupType = Lookup<T>;

  // constructor as a machine
  explicit MPSDiagonal(const Hilbert &hilbert, const json &pars)
      : N_(hilbert.Size()), d_(hilbert.LocalSize()), hilbert_(hilbert) {
    from_json(pars);
  };

  // constructor for use in SBS machine
  MPSDiagonal(const Hilbert &hilbert, const int &N, const int &D,
              const int &symperiod)
      : N_(N),
        d_(hilbert.LocalSize()),
        D_(D),
        symperiod_(symperiod),
        hilbert_(hilbert) {
    Init(false);
  };

  // Auxiliary function that defines the matrices
  void Init(const bool &show_messages) {
    // Initialize parameters
    std::vector<VectorType> pushback_vec;
    VectorType init_mat = VectorType::Zero(D_);
    npar_ = symperiod_ * d_ * D_;

    for (int site = 0; site < symperiod_; site++) {
      W_.push_back(pushback_vec);
      for (int spin = 0; spin < d_; spin++) {
        W_[site].push_back(init_mat);
      }
    }

    // Machine creation messages
    if (show_messages) {
      InfoMessage() << "Periodic diagonal MPS machine with " << N_
                    << " sites created" << std::endl;
      InfoMessage() << "Physical dimension d = " << d_
                    << " and bond dimension D = " << D_ << std::endl;
      if (symperiod_ < N_) {
        InfoMessage() << "Translation invariance is used. Number of "
                         "variational parameters is "
                      << npar_ << " instead of " << npar_ * N_ / symperiod_
                      << std::endl;
      } else {
        InfoMessage() << "Number of variational parameters is " << npar_
                      << std::endl;
      }

      // Initialize map from Hilbert space states to MPS indices
      auto localstates = hilbert_.LocalStates();
      for (int i = 0; i < d_; i++) {
        confindex_[localstates[i]] = i;
      }
    }
  };

  int Npar() const override { return npar_; };

  VectorType GetParameters() override {
    int k = 0;
    VectorType pars(npar_);

    for (int site = 0; site < symperiod_; site++) {
      for (int spin = 0; spin < d_; spin++) {
        for (int i = 0; i < D_; i++) {
          pars(k) = W_[site][spin](i);
          k++;
        }
      }
    }
    return pars;
  };

  void SetParameters(const VectorType &pars) override {
    int k = 0;

    for (int site = 0; site < symperiod_; site++) {
      for (int spin = 0; spin < d_; spin++) {
        for (int i = 0; i < D_; i++) {
          W_[site][spin](i) = pars(k);
          k++;
        }
      }
    }
  };

  // Auxiliary function used for setting initial random parameters and adding
  // identities in every matrix
  inline void SetParametersIdentity(const VectorType &pars) override {
    int k = 0;

    for (int site = 0; site < symperiod_; site++) {
      for (int spin = 0; spin < d_; spin++) {
        for (int i = 0; i < D_; i++) {
          W_[site][spin](i) = T(1, 0) + pars(k);
          k++;
        }
      }
    }
  };

  void InitRandomPars(int seed, double sigma) override {
    VectorType pars(npar_);

    netket::RandomGaussian(pars, seed, sigma);
    SetParametersIdentity(pars);
  };

  int Nvisible() const override { return N_; };

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    // First (left) site
    _InitLookup_check(lt, 0);
    lt.V(0) = W_[0][confindex_[v(0)]];

    // Last (right) site
    _InitLookup_check(lt, 1);
    lt.V(1) = W_[(N_ - 1) % symperiod_][confindex_[v(N_ - 1)]];

    // Rest sites
    for (int i = 2; i < 2 * N_; i += 2) {
      int site = i / 2;
      _InitLookup_check(lt, i);
      lt.V(i) =
          lt.V(i - 2).cwiseProduct(W_[site % symperiod_][confindex_[v(site)]]);

      site = N_ - 1 - site;
      _InitLookup_check(lt, i + 1);
      lt.V(i + 1) =
          W_[site % symperiod_][confindex_[v(site)]].cwiseProduct(lt.V(i - 1));
    }
  };

  // Auxiliary function
  inline void _InitLookup_check(LookupType &lt, int i) {
    if (lt.VectorSize() == i) {
      lt.AddVector(D_);
    } else {
      lt.V(i).resize(D_);
    }
  };

  // Auxiliary function for sorting indeces
  // (copied from stackexchange - original answer by Lukasz Wiklendt)
  inline std::vector<std::size_t> sort_indeces(const std::vector<int> &v) {
    // initialize original index locations
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](std::size_t i1, std::size_t i2) { return v[i1] < v[i2]; });
    return idx;
  };

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    std::size_t nchange = tochange.size();
    if (nchange <= 0) {
      return;
    }
    std::vector<std::size_t> sorted_ind = sort_indeces(tochange);
    int site = tochange[sorted_ind[0]];

    // Update left (site++)
    if (site == 0) {
      lt.V(0) = W_[0][confindex_[newconf[sorted_ind[0]]]];
    } else {
      lt.V(2 * site) =
          lt.V(2 * (site - 1))
              .cwiseProduct(
                  W_[site % symperiod_][confindex_[newconf[sorted_ind[0]]]]);
    }

    for (std::size_t k = 1; k < nchange; k++) {
      for (site = tochange[sorted_ind[k - 1]] + 1;
           site < tochange[sorted_ind[k]]; site++) {
        lt.V(2 * site) =
            lt.V(2 * (site - 1))
                .cwiseProduct(W_[site % symperiod_][confindex_[v(site)]]);
      }
      site = tochange[sorted_ind[k]];
      lt.V(2 * site) =
          lt.V(2 * (site - 1))
              .cwiseProduct(
                  W_[site % symperiod_][confindex_[newconf[sorted_ind[k]]]]);
    }

    for (site = tochange[sorted_ind[nchange - 1]] + 1; site < N_; site++) {
      lt.V(2 * site) =
          lt.V(2 * (site - 1))
              .cwiseProduct(W_[site % symperiod_][confindex_[v(site)]]);
    }

    // Update right (site--)
    site = tochange[sorted_ind[nchange - 1]];
    if (site == N_ - 1) {
      lt.V(1) = W_[(N_ - 1) % symperiod_]
                  [confindex_[newconf[sorted_ind[nchange - 1]]]];
    } else {
      lt.V(2 * (N_ - site) - 1) =
          W_[site % symperiod_][confindex_[newconf[sorted_ind[nchange - 1]]]]
              .cwiseProduct(lt.V(2 * (N_ - site) - 3));
    }

    for (int k = nchange - 2; k >= 0; k--) {
      for (site = tochange[sorted_ind[k + 1]] - 1;
           site > tochange[sorted_ind[k]]; site--) {
        lt.V(2 * (N_ - site) - 1) =
            W_[site % symperiod_][confindex_[v(site)]].cwiseProduct(
                lt.V(2 * (N_ - site) - 3));
      }
      site = tochange[sorted_ind[k]];
      lt.V(2 * (N_ - site) - 1) =
          W_[site % symperiod_][confindex_[newconf[sorted_ind[k]]]]
              .cwiseProduct(lt.V(2 * (N_ - site) - 3));
    }

    for (site = tochange[sorted_ind[0]] - 1; site >= 0; site--) {
      lt.V(2 * (N_ - site) - 1) =
          W_[site % symperiod_][confindex_[v(site)]].cwiseProduct(
              lt.V(2 * (N_ - site) - 3));
    }
  };

  // Auxiliary function that calculates contractions from site1 to site2
  inline VectorType mps_contraction(const Eigen::VectorXd &v, const int &site1,
                                    const int &site2) {
    VectorType c = VectorType::Ones(D_);
    for (int site = site1; site < site2; site++) {
      c = c.cwiseProduct(W_[site % symperiod_][confindex_[v(site)]]);
    }
    return c;
  };

  T LogVal(const Eigen::VectorXd &v) override {
    return std::log(mps_contraction(v, 0, N_).sum());
  };

  T LogVal(const Eigen::VectorXd &v, const LookupType &lt) override {
    return std::log(lt.V(2 * N_ - 2).sum());
  };

  VectorType LogValDiff(
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    int site = 0;

    std::vector<std::size_t> sorted_ind;
    VectorType logvaldiffs = VectorType::Zero(nconn), new_prods(D_);
    StateType current_psi = mps_contraction(v, 0, N_).sum();

    // current_prod calculation only needs to be done once. Fix that
    for (std::size_t k = 0; k < nconn; k++) {
      std::size_t nchange = tochange[k].size();

      if (nchange > 0) {
        sorted_ind = sort_indeces(tochange[k]);
        site = tochange[k][sorted_ind[0]];

        if (site == 0) {
          new_prods = W_[0][confindex_[newconf[k][sorted_ind[0]]]];
        } else {
          new_prods =
              mps_contraction(v, 0, site)
                  .cwiseProduct(W_[site % symperiod_]
                                  [confindex_[newconf[k][sorted_ind[0]]]]);
        }

        for (std::size_t i = 1; i < nchange; i++) {
          site = tochange[k][sorted_ind[i]];
          new_prods = new_prods.cwiseProduct(
              mps_contraction(v, tochange[k][sorted_ind[i - 1]] + 1, site)
                  .cwiseProduct(W_[site % symperiod_]
                                  [confindex_[newconf[k][sorted_ind[i]]]]));
        }
        site = tochange[k][sorted_ind[nchange - 1]];
        if (site < N_ - 1) {
          new_prods = new_prods.cwiseProduct(mps_contraction(v, site + 1, N_));
        }

        logvaldiffs(k) = std::log(new_prods.sum() / current_psi);
      }
    }
    return logvaldiffs;
  };

  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &toflip,
               const std::vector<double> &newconf,
               const LookupType &lt) override {
    const std::size_t nflip = toflip.size();
    if (nflip <= 0) {
      return T(0, 0);
    }
    VectorType new_prod;
    std::vector<std::size_t> sorted_ind = sort_indeces(toflip);
    int site = toflip[sorted_ind[0]];

    if (site == 0) {
      new_prod = W_[0][confindex_[newconf[sorted_ind[0]]]];
    } else {
      new_prod =
          lt.V(2 * (site - 1))
              .cwiseProduct(
                  W_[site % symperiod_][confindex_[newconf[sorted_ind[0]]]]);
    }

    for (std::size_t k = 1; k < nflip; k++) {
      site = toflip[sorted_ind[k]];
      new_prod = new_prod.cwiseProduct(
          mps_contraction(v, toflip[sorted_ind[k - 1]] + 1, site)
              .cwiseProduct(
                  W_[site % symperiod_][confindex_[newconf[sorted_ind[k]]]]));
    }

    site = toflip[sorted_ind[nflip - 1]];
    if (site < N_ - 1) {
      new_prod = new_prod.cwiseProduct(lt.V(2 * (N_ - site) - 3));
    }

    return std::log(new_prod.sum() / lt.V(2 * N_ - 2).sum());
  };

  // Derivative with full calculation
  VectorType DerLog(const Eigen::VectorXd &v) override {
    std::vector<VectorType> left_prods, right_prods;
    VectorType der = VectorType::Zero(npar_), temp_product(D_);

    // Calculate products
    left_prods.push_back(W_[0][confindex_[v(0)]]);
    right_prods.push_back(W_[(N_ - 1) % symperiod_][confindex_[v(N_ - 1)]]);
    for (int site = 1; site < N_ - 1; site++) {
      left_prods.push_back(left_prods[site - 1].cwiseProduct(
          W_[site % symperiod_][confindex_[v(site)]]));
      right_prods.push_back(
          W_[(N_ - 1 - site) % symperiod_][confindex_[v(N_ - 1 - site)]]
              .cwiseProduct(right_prods[site - 1]));
    }
    left_prods.push_back(left_prods[N_ - 2].cwiseProduct(
        W_[(N_ - 1) % symperiod_][confindex_[v(N_ - 1)]]));
    right_prods.push_back(
        W_[0][confindex_[v(0)]].cwiseProduct(right_prods[N_ - 2]));

    der.segment(confindex_[v(0)] * D_, D_) +=
        Eigen::Map<VectorType>((right_prods[N_ - 2]).transpose().data(), D_);
    for (int site = 1; site < N_ - 1; site++) {
      temp_product =
          right_prods[N_ - site - 2].cwiseProduct(left_prods[site - 1]);
      der.segment((d_ * (site % symperiod_) + confindex_[v(site)]) * D_, D_) +=
          Eigen::Map<VectorType>((temp_product).transpose().data(), D_);
    }
    der.segment((d_ * ((N_ - 1) % symperiod_) + confindex_[v(N_ - 1)]) * D_,
                D_) +=
        Eigen::Map<VectorType>((left_prods[N_ - 2]).transpose().data(), D_);

    return der / left_prods[N_ - 1].sum();
  };

  const Hilbert &GetHilbert() const { return hilbert_; };

  // Json functions
  void to_json(json &j) const override {
    j["Machine"]["Name"] = "MPSdiagonal";
    j["Machine"]["Length"] = N_;
    j["Machine"]["BondDim"] = D_;
    j["Machine"]["PhysDim"] = d_;
    j["Machine"]["SymmetryPeriod"] = symperiod_;
    for (int i = 0; i < symperiod_; i++) {
      for (int k = 0; k < d_; k++) {
        j["Machine"]["W" + std::to_string(d_ * i + k)] = W_[i][k];
      }
    }
  };

  void from_json(const json &pars) override {
    if (pars.at("Machine").at("Name") != "MPSdiagonal") {
      throw InvalidInputError("Error while constructing MPS from Json input");
    }

    if (FieldExists(pars["Machine"], "Length")) {
      N_ = pars["Machine"]["Length"];
    }
    if (N_ != hilbert_.Size()) {
      throw InvalidInputError(
          "Number of spins is incompatible with given Hilbert space");
    }

    if (FieldExists(pars["Machine"], "PhysDim")) {
      d_ = pars["Machine"]["PhysDim"];
    }
    if (d_ != hilbert_.LocalSize()) {
      throw InvalidInputError(
          "Number of spins is incompatible with given Hilbert space");
    }

    if (FieldExists(pars["Machine"], "BondDim")) {
      D_ = pars["Machine"]["BondDim"];
    } else {
      throw InvalidInputError("Unspecified bond dimension");
    }

    if (FieldExists(pars["Machine"], "SymmetryPeriod")) {
      symperiod_ = pars["Machine"]["SymmetryPeriod"];
    } else {
      symperiod_ = N_;
    }

    Init(true);

    // Loading parameters, if defined in the input
    from_jsonWeights(pars["Machine"]);
  };

  // Used in SBS too
  inline void from_jsonWeights(const json &pars) override {
    for (int i = 0; i < symperiod_; i++) {
      for (int k = 0; k < d_; k++) {
        if (FieldExists(pars, "W" + std::to_string(d_ * i + k))) {
          W_[i][k] = pars["W" + std::to_string(d_ * i + k)];
        }
      }
    }
  };

  // ###################################### //
  // ##### Functions for SBS use only ##### //
  // ###################################### //
  // We treat SBS differently for efficiency:
  // Otherwise we would have to define a different confindex_
  // for each MPS string in the SBS

  void InitLookup(const std::vector<int> &v, LookupType &lt,
                  const int &start_ind) override {
    // First (left) site
    _InitLookup_check(lt, start_ind);
    lt.V(start_ind) = W_[0][v[0]];

    // Last (right) site
    _InitLookup_check(lt, start_ind + 1);
    lt.V(start_ind + 1) = W_[(N_ - 1) % symperiod_][v[N_ - 1]];

    // Rest sites
    for (int i = 2; i < 2 * N_; i += 2) {
      _InitLookup_check(lt, start_ind + i);
      int site = i / 2;
      lt.V(start_ind + i) = lt.V(start_ind + i - 2)
                                .cwiseProduct(W_[(site % symperiod_)][v[site]]);

      _InitLookup_check(lt, start_ind + i + 1);
      site = N_ - 1 - site;
      lt.V(start_ind + i + 1) =
          W_[site % symperiod_][v[site]].cwiseProduct(lt.V(start_ind + i - 1));
    }
  };

  void UpdateLookup(const std::vector<int> &v, const std::vector<int> &tochange,
                    const std::vector<int> &newconf, LookupType &lt,
                    const int &start_ind) override {
    std::size_t nchange = tochange.size();
    if (nchange <= 0) {
      return;
    }
    std::vector<std::size_t> sorted_ind = sort_indeces(tochange);
    int site = tochange[sorted_ind[0]];

    // Update left (site++)
    if (site == 0) {
      lt.V(start_ind) = W_[0][newconf[sorted_ind[0]]];
    } else {
      lt.V(start_ind + 2 * site) =
          lt.V(start_ind + 2 * (site - 1))
              .cwiseProduct(W_[site % symperiod_][newconf[sorted_ind[0]]]);
    }

    for (std::size_t k = 1; k < nchange; k++) {
      for (site = tochange[sorted_ind[k - 1]] + 1;
           site < tochange[sorted_ind[k]]; site++) {
        lt.V(start_ind + 2 * site) =
            lt.V(start_ind + 2 * (site - 1))
                .cwiseProduct(W_[site % symperiod_][v[site]]);
      }
      site = tochange[sorted_ind[k]];
      lt.V(start_ind + 2 * site) =
          lt.V(start_ind + 2 * (site - 1))
              .cwiseProduct(W_[site % symperiod_][newconf[sorted_ind[k]]]);
    }

    for (site = tochange[sorted_ind[nchange - 1]] + 1; site < N_; site++) {
      lt.V(start_ind + 2 * site) =
          lt.V(start_ind + 2 * (site - 1))
              .cwiseProduct(W_[site % symperiod_][v[site]]);
    }

    // Update right (site--)
    site = tochange[sorted_ind[nchange - 1]];
    if (site == N_ - 1) {
      lt.V(start_ind + 1) =
          W_[(N_ - 1) % symperiod_][newconf[sorted_ind[nchange - 1]]];
    } else {
      lt.V(start_ind + 2 * (N_ - site) - 1) =
          W_[site % symperiod_][newconf[sorted_ind[nchange - 1]]].cwiseProduct(
              lt.V(start_ind + 2 * (N_ - site) - 3));
    }

    for (int k = nchange - 2; k >= 0; k--) {
      for (site = tochange[sorted_ind[k + 1]] - 1;
           site > tochange[sorted_ind[k]]; site--) {
        lt.V(start_ind + 2 * (N_ - site) - 1) =
            W_[site % symperiod_][v[site]].cwiseProduct(
                lt.V(start_ind + 2 * (N_ - site) - 3));
      }
      site = tochange[sorted_ind[k]];
      lt.V(start_ind + 2 * (N_ - site) - 1) =
          W_[site % symperiod_][newconf[sorted_ind[k]]].cwiseProduct(
              lt.V(start_ind + 2 * (N_ - site) - 3));
    }

    for (site = tochange[sorted_ind[0]] - 1; site >= 0; site--) {
      lt.V(start_ind + 2 * (N_ - site) - 1) =
          W_[site % symperiod_][v[site]].cwiseProduct(
              lt.V(start_ind + 2 * (N_ - site) - 3));
    }
  };

  // Auxilliary function that calculates MPS contractions from site1 to site2
  inline VectorType mps_contraction(const std::vector<int> &v, const int &site1,
                                    const int &site2) {
    VectorType c = VectorType::Ones(D_);
    for (int site = site1; site < site2; site++) {
      c = c.cwiseProduct(W_[site % symperiod_][v[site]]);
    }
    return c;
  };

  T LogVal(const std::vector<int> &v) override {
    return std::log(mps_contraction(v, 0, N_).sum());
  };

  inline T LogVal(const LookupType &lt, const int &start_ind) override {
    return std::log(lt.V(start_ind + 2 * N_ - 2).sum());
  };

  T LogValDiff(const std::vector<int> &v, const std::vector<int> &toflip,
               const std::vector<int> &newconf) override {
    const std::size_t nflip = toflip.size();
    if (nflip <= 0) {
      return T(0, 0);
    }

    std::vector<std::size_t> sorted_ind = sort_indeces(toflip);
    StateType current_psi = mps_contraction(v, 0, N_).sum();
    VectorType new_prods(D_);

    if (toflip[sorted_ind[0]] == 0) {
      new_prods = W_[0][newconf[sorted_ind[0]]];
    } else {
      new_prods = mps_contraction(v, 0, toflip[sorted_ind[0]])
                      .cwiseProduct(W_[toflip[sorted_ind[0]] % symperiod_]
                                      [newconf[sorted_ind[0]]]);
    }
    for (std::size_t i = 1; i < nflip; i++) {
      new_prods = new_prods.cwiseProduct(
          mps_contraction(v, toflip[sorted_ind[i - 1]] + 1,
                          toflip[sorted_ind[i]])
              .cwiseProduct(W_[toflip[sorted_ind[i]] % symperiod_]
                              [newconf[sorted_ind[i]]]));
    }
    if (toflip[sorted_ind[nflip - 1]] < N_ - 1) {
      new_prods = new_prods.cwiseProduct(
          mps_contraction(v, toflip[sorted_ind[nflip - 1]] + 1, N_));
    }
    return std::log(new_prods.sum() / current_psi);
  };

  T LogValDiff(const std::vector<int> &v, const std::vector<int> &toflip,
               const std::vector<int> &newconf, const LookupType &lt,
               const int &start_ind) override {
    const std::size_t nflip = toflip.size();
    if (nflip <= 0) {
      return T(0, 0);
    }

    std::vector<std::size_t> sorted_ind = sort_indeces(toflip);
    VectorType new_prods(D_);
    int site = toflip[sorted_ind[0]];

    if (site == 0) {
      new_prods = W_[0][newconf[sorted_ind[0]]];
    } else {
      new_prods =
          lt.V(start_ind + 2 * (site - 1))
              .cwiseProduct(W_[site % symperiod_][newconf[sorted_ind[0]]]);
    }

    for (std::size_t i = 1; i < nflip; i++) {
      site = toflip[sorted_ind[i]];
      new_prods = new_prods.cwiseProduct(
          mps_contraction(v, toflip[sorted_ind[i - 1]] + 1, site)
              .cwiseProduct(W_[site % symperiod_][newconf[sorted_ind[i]]]));
    }

    site = toflip[sorted_ind[nflip - 1]];
    if (site < N_ - 1) {
      new_prods = new_prods.cwiseProduct(lt.V(start_ind + 2 * (N_ - site) - 3));
    }
    return std::log(new_prods.sum() / lt.V(start_ind + 2 * N_ - 2).sum());
  };

  T FastLogValDiff(const std::vector<int> &toflip,
                   const std::vector<int> &newconf, const LookupType &lt,
                   const int &start_ind) override {
    const std::size_t nflip = toflip.size();
    if (nflip <= 0) {
      return T(0, 0);
    }

    MatrixType new_prods(D_, D_);
    int site = toflip[0];

    if (site == 0) {
      new_prods = W_[0][newconf[0]].cwiseProduct(lt.V(start_ind + 2 * N_ - 3));
    } else if (site == N_ - 1) {
      new_prods = lt.V(start_ind + 2 * N_ - 4)
                      .cwiseProduct(W_[(N_ - 1) % symperiod_][newconf[0]]);
    } else {
      new_prods = (lt.V(start_ind + 2 * (site - 1))
                       .cwiseProduct(W_[site % symperiod_][newconf[0]]))
                      .cwiseProduct(lt.V(start_ind + 2 * (N_ - site) - 3));
    }
    return std::log(new_prods.sum() / lt.V(start_ind + 2 * N_ - 2).sum());
  };

  VectorType DerLog(const std::vector<int> &v) override {
    VectorType temp_product(D_);
    std::vector<VectorType> left_prods, right_prods;
    VectorType der = VectorType::Zero(npar_);

    // Calculate products
    left_prods.push_back(W_[0][v[0]]);
    right_prods.push_back(W_[(N_ - 1) % symperiod_][v[N_ - 1]]);
    for (int site = 1; site < N_ - 1; site++) {
      left_prods.push_back(
          left_prods[site - 1].cwiseProduct(W_[site][v[site]]));
      right_prods.push_back(
          W_[(N_ - 1 - site) % symperiod_][v[N_ - 1 - site]].cwiseProduct(
              right_prods[site - 1]));
    }
    left_prods.push_back(
        left_prods[N_ - 2].cwiseProduct(W_[(N_ - 1) % symperiod_][v[N_ - 1]]));
    right_prods.push_back(W_[0][v[0]].cwiseProduct(right_prods[N_ - 2]));

    der.segment(v[0] * D_, D_) +=
        Eigen::Map<VectorType>((right_prods[N_ - 2]).transpose().data(), D_);
    for (int site = 1; site < N_ - 1; site++) {
      temp_product =
          right_prods[N_ - site - 2].cwiseProduct(left_prods[site - 1]);
      der.segment((d_ * (site % symperiod_) + v[site]) * D_, D_) +=
          Eigen::Map<VectorType>((temp_product).transpose().data(), D_);
    }
    der.segment((d_ * ((N_ - 1) % symperiod_) + v[N_ - 1]) * D_, D_) +=
        Eigen::Map<VectorType>((left_prods[N_ - 2]).transpose().data(), D_);

    return der / left_prods[N_ - 1].sum();
  };

  void to_json_strings(json &j,
                       const std::vector<int> &string2site) const override {
    json stringpar;

    stringpar["Length"] = N_;
    stringpar["BondDim"] = D_;
    stringpar["SymmetryPeriod"] = symperiod_;
    stringpar["Diagonal"] = true;
    stringpar["SiteNumbers"] = string2site;
    for (int i = 0; i < symperiod_; i++) {
      for (int k = 0; k < d_; k++) {
        stringpar["W" + std::to_string(d_ * i + k)] = W_[i][k];
      }
    }

    j["Machine"]["Strings"].push_back(stringpar);
  };
};

}  // namespace netket

#endif
