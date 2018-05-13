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

#ifndef NETKET_METROPOLISEXCHANGEPT_HH
#define NETKET_METROPOLISEXCHANGEPT_HH

#include "Parallel/parallel.hh"
#include "abstract_sampler.hh"
#include <Eigen/Dense>
#include <iostream>
#include <mpi.h>
#include <random>

namespace netket {

// Metropolis sampling generating local exchanges
// Parallel tempering is also used
template <class WfType>
class MetropolisExchangePt : public AbstractSampler<WfType> {

  WfType &psi_;
  const Hilbert &hilbert_;

  // number of visible units
  const int nv_;

  netket::default_random_engine rgen_;

  // states of visible units
  // for each sampled temperature
  std::vector<Eigen::VectorXd> v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // clusters to do updates
  std::vector<std::vector<int>> clusters_;

  // Look-up tables
  std::vector<typename WfType::LookupType> lt_;

  bool do_sum_constraint_;
  int sum_constraint_;

  const int nrep_;

  std::vector<double> beta_;

public:
  template <class G>
  MetropolisExchangePt(G &graph, WfType &psi, int nrep, int dmax = 1)
      : psi_(psi), hilbert_(psi.GetHilbert()), nv_(hilbert_.Size()),
        nrep_(nrep) {
    Init(graph, dmax);
  }

  // Json constructor
  MetropolisExchangePt(Graph &graph, WfType &psi, const json &pars)
      : psi_(psi), hilbert_(psi.GetHilbert()), nv_(hilbert_.Size()),
        nrep_(FieldVal(pars["Sampler"], "Nreplicas")) {

    int dmax = FieldOrDefaultVal(pars["Sampler"], "Dmax", 1);
    Init(graph, dmax);
  }

  template <class G> void Init(G &graph, int dmax) {
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    do_sum_constraint_ = false;

    v_.resize(nrep_);
    for (int i = 0; i < nrep_; i++) {
      v_[i].resize(nv_);
    }

    for (int i = 0; i < nrep_; i++) {
      beta_.push_back(1. - double(i) / double(nrep_));
    }

    lt_.resize(nrep_);

    accept_.resize(2 * nrep_);
    moves_.resize(2 * nrep_);

    GenerateClusters(graph, dmax);

    Seed();

    Reset(true);

    if (mynode_ == 0) {
      std::cout << "# Metropolis sampler with parallel tempering is ready "
                << std::endl;
      std::cout << "# " << nrep_ << " replicas are being used" << std::endl;
      std::cout << "# " << dmax << " is the maximum distance for exchanges"
                << std::endl;
    }
  }

  template <class Graph> void GenerateClusters(Graph &graph, int dmax) {
    auto dist = graph.Distances();

    assert(int(dist.size()) == nv_);

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nv_; j++) {
        if (dist[i][j] <= dmax && i != j) {
          clusters_.push_back({i, j});
        }
      }
    }
  }

  void Seed(int baseseed = 0) {
    std::random_device rd;
    std::vector<int> seeds(totalnodes_);

    if (mynode_ == 0) {
      for (int i = 0; i < totalnodes_; i++) {
        seeds[i] = rd() + baseseed;
      }
    }

    SendToAll(seeds);

    rgen_.seed(seeds[mynode_]);
  }

  void Reset(bool initrandom = false) {
    if (initrandom) {
      for (int i = 0; i < nrep_; i++) {
        hilbert_.RandomVals(v_[i], rgen_);
      }
    }

    for (int i = 0; i < nrep_; i++) {
      psi_.InitLookup(v_[i], lt_[i]);
    }

    accept_ = Eigen::VectorXd::Zero(2 * nrep_);
    moves_ = Eigen::VectorXd::Zero(2 * nrep_);
  }

  // Exchange sweep at given temperature
  void LocalExchangeSweep(int rep) {

    std::vector<int> tochange(2);
    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distcl(0, clusters_.size() - 1);

    std::vector<double> newconf(2);

    for (int i = 0; i < nv_; i++) {

      int rcl = distcl(rgen_);
      assert(rcl < int(clusters_.size()));
      int si = clusters_[rcl][0];
      int sj = clusters_[rcl][1];

      assert(si < nv_ && sj < nv_);

      if (std::abs(v_[rep](si) - v_[rep](sj)) >
          std::numeric_limits<double>::epsilon()) {

        tochange = clusters_[rcl];
        newconf[0] = v_[rep](sj);
        newconf[1] = v_[rep](si);

        double ratio = std::norm(
            std::exp(beta_[rep] *
                     psi_.LogValDiff(v_[rep], tochange, newconf, lt_[rep])));

        if (ratio > distu(rgen_)) {
          accept_(rep) += 1;
          psi_.UpdateLookup(v_[rep], tochange, newconf, lt_[rep]);
          hilbert_.UpdateConf(v_[rep], tochange, newconf);
        }
      }

      moves_(rep) += 1;
    }
  }

  void Sweep() {

    // First we do local exchange sweeps
    for (int i = 0; i < nrep_; i++) {
      LocalExchangeSweep(i);
    }

    // Tempearture exchanges
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int r = 1; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(rgen_)) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }

    for (int r = 2; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(rgen_)) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }
  }

  // computes the probability to exchange two replicas
  double ExchangeProb(int r1, int r2) {
    const double lf1 = 2 * realpart(psi_.LogVal(v_[r1], lt_[r1]));
    const double lf2 = 2 * realpart(psi_.LogVal(v_[r2], lt_[r2]));

    return std::exp((beta_[r1] - beta_[r2]) * (lf2 - lf1));
  }

  void Exchange(int r1, int r2) {
    std::swap(v_[r1], v_[r2]);
    std::swap(lt_[r1], lt_[r2]);
  }

  Eigen::VectorXd Visible() { return v_[0]; }

  void SetVisible(const Eigen::VectorXd &v) { v_[0] = v; }

  WfType &Psi() { return psi_; }

  Hilbert &HilbSpace() const { return hilbert_; }

  Eigen::VectorXd Acceptance() const {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < acc.size(); i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }

  inline double realpart(const std::complex<double> &val) const {
    return val.real();
  }
  inline double realpart(const double &val) const { return val; }
};

} // namespace netket

#endif
