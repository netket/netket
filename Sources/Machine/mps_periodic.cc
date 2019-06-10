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

#include "Machine/mps_periodic.hpp"

#include <set>

#include "Utils/json_utils.hpp"
#include "Utils/messages.hpp"

namespace netket {

MPSPeriodic::MPSPeriodic(std::shared_ptr<const AbstractHilbert> hilbert,
                         int bond_dim, bool diag, int symperiod)
    : AbstractMachine(hilbert),
      N_{hilbert->Size()},
      d_{hilbert->LocalSize()},
      D_{bond_dim},
      symperiod_{symperiod},
      is_diag_{diag} {
  if (symperiod_ == -1) {
    symperiod_ = N_;
  }
  Init();
}

MPSPeriodic::MatrixType MPSPeriodic::prod(const MatrixType &m1,
                                          const MatrixType &m2) const {
  if (is_diag_) {
    return m1.cwiseProduct(m2);
  }
  return m1 * m2;
}

Complex MPSPeriodic::trace(const MatrixType &m) const {
  if (is_diag_) {
    return m.sum();
  }
  return m.trace();
}

void MPSPeriodic::setparamsident(MatrixType &m, VectorConstRefType pars) const {
  if (is_diag_) {
    for (int i = 0; i < D_; i++) {
      m(i, 0) = Complex(1, 0) + pars(i);
    }
  } else {
    for (int i = 0; i < D_; i++) {
      for (int j = 0; j < D_; j++) {
        m(i, j) = pars(i * D_ + j);
        if (i == j) {
          m(i, j) += Complex(1, 0);
        }
      }
    }
  }
}

void MPSPeriodic::Init() {
  // Initialize parameters
  std::vector<MatrixType> pushback_vec;
  if (is_diag_) {
    Dsec_ = 1;
    identity_mat_ = MatrixType::Ones(D_, 1);
  } else {
    Dsec_ = D_;
    identity_mat_ = MatrixType::Identity(D_, D_);
  }
  MatrixType init_mat = MatrixType::Zero(D_, Dsec_);
  Dsq_ = D_ * Dsec_;
  npar_ = symperiod_ * d_ * Dsq_;

  for (int site = 0; site < symperiod_; site++) {
    W_.push_back(pushback_vec);
    for (int spin = 0; spin < d_; spin++) {
      W_[site].push_back(init_mat);
    }
  }

  // Initialize tree parameters
  InitTree();

  // Machine creation messages
  if (is_diag_) {
    InfoMessage() << "Periodic diagonal MPS machine with " << N_
                  << " sites created" << std::endl;
  } else {
    InfoMessage() << "Periodic MPS machine with " << N_ << " sites created"
                  << std::endl;
  }
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
  auto localstates = GetHilbert().LocalStates();
  for (int i = 0; i < d_; i++) {
    confindex_[localstates[i]] = i;
  }
}

void MPSPeriodic::InitTree() {
  // Initializes vectors used for tree look up tables
  // leaves_of_site_ and leaf_contractions_

  std::vector<int> two_vector(2), empty_vector, level_start;
  std::vector<std::vector<int>> above;
  int level = 1, available_ind = 0, available_level = 0;
  bool available = false;

  level_start.push_back(0);
  level_start.push_back(N_);

  int length = level_start[level] - level_start[level - 1];
  while (length > 1 or available) {
    above.push_back(empty_vector);
    // Iterate level-1
    for (int i = 0; i < length - 1; i += 2) {
      // Construct above for level-1
      above.back().push_back(level_start[level] + i / 2);
      above.back().push_back(level_start[level] + i / 2);
      // Construct leaf_contractions_ for level
      two_vector[0] = level_start[level - 1] + i;
      two_vector[1] = level_start[level - 1] + i + 1;
      leaf_contractions_.push_back(two_vector);
    }
    if (length % 2 == 1) {
      if (available) {
        // Connect the two odd leaves
        above.back().push_back(level_start[level] + (length - 1) / 2);
        above[available_level].push_back(level_start[level] + (length - 1) / 2);

        two_vector[0] = level_start[level] - 1;
        two_vector[1] = available_ind;
        leaf_contractions_.push_back(two_vector);

        level_start.push_back(level_start.back() + length / 2 + 1);
        available = false;
      } else {
        available = true;
        available_ind = level_start[level] - 1;
        available_level = level - 1;
        level_start.push_back(level_start.back() + length / 2);
      }
    } else {
      level_start.push_back(level_start.back() + length / 2);
    }
    level++;
    length = level_start[level] - level_start[level - 1];
  }
  Nleaves_ = level_start.back();

  // Flatten above vector
  std::vector<int> flat_above;
  for (std::size_t l = 0; l < above.size(); l++) {
    for (std::size_t k = 0; k < above[l].size(); k++) {
      flat_above.push_back(above[l][k]);
    }
  }

  // Create leaves_of_site_ from above vector
  for (int i = 0; i < N_; i++) {
    std::vector<int> leaves;
    leaves.push_back(flat_above[i]);
    while (flat_above[leaves.back()] < Nleaves_ - 1) {
      leaves.push_back(flat_above[leaves.back()]);
    }
    leaves.push_back(Nleaves_ - 1);
    leaves_of_site_.push_back(leaves);
    leaves.clear();
  }
  Nleaves_ += -N_;
}

int MPSPeriodic::Npar() const { return npar_; }
int MPSPeriodic::Nvisible() const { return N_; }

MPSPeriodic::VectorType MPSPeriodic::GetParameters() {
  int k = 0;
  VectorType pars(npar_);

  for (int site = 0; site < symperiod_; site++) {
    for (int spin = 0; spin < d_; spin++) {
      for (int i = 0; i < D_; i++) {
        for (int j = 0; j < Dsec_; j++) {
          pars(k) = W_[site][spin](i, j);
          k++;
        }
      }
    }
  }
  return pars;
}

void MPSPeriodic::SetParameters(VectorConstRefType pars) {
  int k = 0;

  for (int site = 0; site < symperiod_; site++) {
    for (int spin = 0; spin < d_; spin++) {
      for (int i = 0; i < D_; i++) {
        for (int j = 0; j < Dsec_; j++) {
          W_[site][spin](i, j) = pars(k);
          k++;
        }
      }
    }
  }
}

// Auxiliary function used for setting initial random parameters and adding
// identities in every matrix
void MPSPeriodic::SetParametersIdentity(VectorConstRefType pars) {
  int k = 0;
  for (int site = 0; site < symperiod_; site++) {
    for (int spin = 0; spin < d_; spin++) {
      setparamsident(W_[site][spin], pars.segment(k, Dsq_));
      k += Dsq_;
    }
  }
}

void MPSPeriodic::InitRandomPars(int seed, double sigma) {
  VectorType pars(npar_);

  netket::RandomGaussian(pars, seed, sigma);
  SetParametersIdentity(pars);
}

void MPSPeriodic::InitLookup(VisibleConstType v, LookupType &lt) {
  for (int k = 0; k < Nleaves_; k++) {
    _InitLookup_check(lt, k);
    if (leaf_contractions_[k][0] < N_) {
      if (leaf_contractions_[k][1] < N_) {
        lt.M(k) = prod(W_[leaf_contractions_[k][0] % symperiod_]
                         [confindex_[v(leaf_contractions_[k][0])]],
                       W_[leaf_contractions_[k][1] % symperiod_]
                         [confindex_[v(leaf_contractions_[k][1])]]);
      } else {
        lt.M(k) = prod(W_[leaf_contractions_[k][0] % symperiod_]
                         [confindex_[v(leaf_contractions_[k][0])]],
                       lt.M(leaf_contractions_[k][1] - N_));
      }

    } else {
      if (leaf_contractions_[k][1] < N_) {
        lt.M(k) = prod(lt.M(leaf_contractions_[k][0] - N_),
                       W_[leaf_contractions_[k][1] % symperiod_]
                         [confindex_[v(leaf_contractions_[k][1])]]);
      } else {
        lt.M(k) = prod(lt.M(leaf_contractions_[k][0] - N_),
                       lt.M(leaf_contractions_[k][1] - N_));
      }
    }
  }
}

// Auxiliary function
void MPSPeriodic::_InitLookup_check(LookupType &lt, int i) {
  if (lt.MatrixSize() == i) {
    lt.AddMatrix(D_, Dsec_);
  } else {
    lt.M(i).resize(D_, Dsec_);
  }
}

// Auxiliary function for sorting indeces
// (copied from stackexchange - original answer by Lukasz Wiklendt)
std::vector<std::size_t> MPSPeriodic::sort_indeces(const std::vector<int> &v) {
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](std::size_t i1, std::size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

void MPSPeriodic::UpdateLookup(VisibleConstType v,
                               const std::vector<int> &tochange,
                               const std::vector<double> &newconf,
                               LookupType &lt) {
  std::size_t nchange = tochange.size();
  if (nchange <= 0) {
    return;
  }

  MatrixType empty_matrix = MatrixType::Zero(D_, Dsec_);
  std::vector<std::size_t> sorted_ind = sort_indeces(tochange);

  std::set<int> leaves2update;
  std::size_t set_len = 0;
  std::map<int, MatrixType *> base_mat;

  for (std::size_t k = 0; k < nchange; k++) {
    int site = tochange[sorted_ind[k]];
    // Add changed visible matrices to map
    base_mat[site] =
        &(W_[site % symperiod_][confindex_[newconf[sorted_ind[k]]]]);
    // Add the rest of matrices that affects
    for (std::size_t l = 0; l < leaves_of_site_[site].size(); l++) {
      int leaf = leaves_of_site_[site][l];
      leaves2update.insert(leaf);
      if (leaves2update.size() > set_len) {
        set_len++;
        for (int i = 0; i < 2; i++) {
          int lc = leaf_contractions_[leaf - N_][i];
          if (lc < N_) {
            if (base_mat.count(lc) == 0) {
              base_mat[lc] = &(W_[lc % symperiod_][confindex_[v(lc)]]);
            }
          } else {
            base_mat[lc] = &(lt.M(lc - N_));
          }
        }
      }
    }
  }
  for (auto leaf : leaves2update) {
    lt.M(leaf - N_) = prod(*(base_mat[leaf_contractions_[leaf - N_][0]]),
                           *(base_mat[leaf_contractions_[leaf - N_][1]]));
  }
}

// Auxiliary function that calculates contractions from site1 to site2
MPSPeriodic::MatrixType MPSPeriodic::mps_contraction(VisibleConstType v,
                                                     const int &site1,
                                                     const int &site2) {
  MatrixType c = identity_mat_;
  for (int site = site1; site < site2; site++) {
    c = prod(c, W_[site % symperiod_][confindex_[v(site)]]);
  }
  return c;
}

Complex MPSPeriodic::LogVal(VisibleConstType v) {
  return std::log(trace(mps_contraction(v, 0, N_)));
}

Complex MPSPeriodic::LogVal(VisibleConstType /* v */, const LookupType &lt) {
  return std::log(trace(lt.M(Nleaves_ - 1)));
}

MPSPeriodic::VectorType MPSPeriodic::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  const std::size_t nconn = tochange.size();

  std::vector<std::size_t> sorted_ind;
  VectorType logvaldiffs = VectorType::Zero(nconn);
  Complex current_psi = trace(mps_contraction(v, 0, N_));
  MatrixType new_prods(D_, Dsec_);

  for (std::size_t k = 0; k < nconn; k++) {
    std::size_t nchange = tochange[k].size();
    if (nchange > 0) {
      sorted_ind = sort_indeces(tochange[k]);
      int site = tochange[k][sorted_ind[0]];

      if (site == 0) {
        new_prods = W_[0][confindex_[newconf[k][sorted_ind[0]]]];
      } else {
        new_prods =
            prod(mps_contraction(v, 0, site),
                 W_[site % symperiod_][confindex_[newconf[k][sorted_ind[0]]]]);
      }

      for (std::size_t i = 1; i < nchange; i++) {
        site = tochange[k][sorted_ind[i]];
        new_prods = prod(
            new_prods,
            prod(mps_contraction(v, tochange[k][sorted_ind[i - 1]] + 1, site),
                 W_[site % symperiod_][confindex_[newconf[k][sorted_ind[i]]]]));
      }
      site = tochange[k][sorted_ind[nchange - 1]];
      if (site < N_ - 1) {
        new_prods = prod(new_prods, mps_contraction(v, site + 1, N_));
      }
      logvaldiffs(k) = std::log(trace(new_prods) / current_psi);
    }
  }
  return logvaldiffs;
}

Complex MPSPeriodic::LogValDiff(VisibleConstType v,
                                const std::vector<int> &toflip,
                                const std::vector<double> &newconf,
                                const LookupType &lt) {
  // Assumes number of levels > 1 (?)
  std::size_t nflip = toflip.size();
  if (nflip <= 0) {
    return Complex(0, 0);
  }

  MatrixType empty_matrix = MatrixType::Zero(D_, Dsec_);
  std::vector<std::size_t> sorted_ind = sort_indeces(toflip);

  std::set<int> leaves2update;
  std::map<int, MatrixType> ltpM;
  std::size_t set_len = 0;

  std::vector<bool> two_vector(2, false);
  std::map<int, std::vector<bool>> contractions_change;

  for (std::size_t k = 0; k < nflip; k++) {
    int site = toflip[sorted_ind[k]];
    // Add changed visible matrices to map
    ltpM[site] = W_[site % symperiod_][confindex_[newconf[sorted_ind[k]]]];

    // Add the rest of matrices that affects
    for (std::size_t l = 0; l < leaves_of_site_[site].size(); l++) {
      int leaf = leaves_of_site_[site][l];
      leaves2update.insert(leaf);
      if (leaves2update.size() > set_len) {
        set_len++;
        ltpM[leaf] = empty_matrix;
      }
      // Check the contractions of the new matrix (if they change)
      for (int i = 0; i < 2; i++) {
        if (std::find(toflip.begin(), toflip.end(),
                      leaf_contractions_[leaf - N_][i]) != toflip.end()) {
          two_vector[i] = true;
        } else if (leaves2update.find(leaf_contractions_[leaf - N_][i]) !=
                   leaves2update.end()) {
          two_vector[i] = true;
        }
      }
      contractions_change[leaf] = two_vector;
      two_vector[0] = false;
      two_vector[1] = false;
    }
  }

  // Calculate products
  for (auto leaf : leaves2update) {
    std::vector<MatrixType> m(2);
    for (int i = 0; i < 2; i++) {
      int lc = leaf_contractions_[leaf - N_][i];
      if (contractions_change[leaf][i]) {
        m[i] = ltpM[lc];
      } else {
        if (lc < N_) {
          m[i] = W_[lc % symperiod_][confindex_[v(lc)]];
        } else {
          m[i] = lt.M(lc - N_);
        }
      }
    }
    ltpM[leaf] = prod(m[0], m[1]);
  }
  return std::log(trace(ltpM[Nleaves_ + N_ - 1]) / trace(lt.M(Nleaves_ - 1)));
}

// Derivative with full calculation
MPSPeriodic::VectorType MPSPeriodic::DerLog(VisibleConstType v) {
  MatrixType temp_product(D_, Dsec_);
  std::vector<MatrixType> left_prods, right_prods;
  VectorType der = VectorType::Zero(npar_);

  // Calculate products
  left_prods.push_back(W_[0][confindex_[v(0)]]);
  right_prods.push_back(W_[(N_ - 1) % symperiod_][confindex_[v(N_ - 1)]]);
  for (int site = 1; site < N_ - 1; site++) {
    left_prods.push_back(
        prod(left_prods[site - 1], W_[site % symperiod_][confindex_[v(site)]]));
    right_prods.push_back(
        prod(W_[(N_ - 1 - site) % symperiod_][confindex_[v(N_ - 1 - site)]],
             right_prods[site - 1]));
  }
  left_prods.push_back(prod(left_prods[N_ - 2],
                            W_[(N_ - 1) % symperiod_][confindex_[v(N_ - 1)]]));
  right_prods.push_back(prod(W_[0][confindex_[v(0)]], right_prods[N_ - 2]));

  der.segment(confindex_[v(0)] * Dsq_, Dsq_) +=
      Eigen::Map<VectorType>(right_prods[N_ - 2].transpose().data(), Dsq_);
  for (int site = 1; site < N_ - 1; site++) {
    temp_product = prod(right_prods[N_ - site - 2], left_prods[site - 1]);
    der.segment((d_ * (site % symperiod_) + confindex_[v(site)]) * Dsq_,
                Dsq_) +=
        Eigen::Map<VectorType>(temp_product.transpose().data(), Dsq_);
  }
  der.segment((d_ * ((N_ - 1) % symperiod_) + confindex_[v(N_ - 1)]) * Dsq_,
              Dsq_) +=
      Eigen::Map<VectorType>(left_prods[N_ - 2].transpose().data(), Dsq_);

  return der / trace(left_prods[N_ - 1]);
}

// Json functions
void MPSPeriodic::to_json(json &j) const {
  j["Name"] = "MPSperiodic";
  j["Length"] = N_;
  j["BondDim"] = D_;
  j["PhysDim"] = d_;
  j["Diagonal"] = is_diag_;
  j["SymmetryPeriod"] = symperiod_;
  for (int i = 0; i < symperiod_; i++) {
    for (int k = 0; k < d_; k++) {
      j["W" + std::to_string(d_ * i + k)] = W_[i][k];
    }
  }
}

void MPSPeriodic::from_json(const json &pars) {
  if (pars.at("Name") != "MPSperiodic") {
    throw InvalidInputError("Error while constructing MPS from Json input");
  }

  if (FieldExists(pars, "Length")) {
    N_ = pars["Length"];
  }
  if (N_ != GetHilbert().Size()) {
    throw InvalidInputError(
        "Number of spins is incompatible with given Hilbert space");
  }

  if (FieldExists(pars, "PhysDim")) {
    d_ = pars["PhysDim"];
  }
  if (d_ != GetHilbert().LocalSize()) {
    throw InvalidInputError(
        "Number of spins is incompatible with given Hilbert space");
  }

  if (FieldExists(pars, "BondDim")) {
    D_ = pars["BondDim"];
  } else {
    throw InvalidInputError("Unspecified bond dimension");
  }

  if (FieldExists(pars, "Diagonal")) {
    is_diag_ = pars["Diagonal"];
  } else {
    throw InvalidInputError("Unspecified whether the MPS is diagonal");
  }

  if (FieldExists(pars, "SymmetryPeriod")) {
    symperiod_ = pars["SymmetryPeriod"];
  } else {
    // Default is symperiod = N, resp. no translational symmetry
    symperiod_ = N_;
  }

  Init();

  // Loading parameters, if defined in the input
  from_jsonWeights(pars);
}

void MPSPeriodic::from_jsonWeights(const json &pars) {
  for (int i = 0; i < symperiod_; i++) {
    for (int k = 0; k < d_; k++) {
      if (FieldExists(pars, "W" + std::to_string(d_ * i + k))) {
        W_[i][k] = pars["W" + std::to_string(d_ * i + k)];
      }
    }
  }
}

}  // namespace netket
