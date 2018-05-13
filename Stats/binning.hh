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

#ifndef NETKET_BINNING_HH
#define NETKET_BINNING_HH

#include <Eigen/Dense>
#include <valarray>
#include <vector>

namespace netket {

template <class T> class Binning {

  using StatType = OnlineStat<T>;
  using DataType = T;

  // target number of bins
  // the actual number of bins used for the estimates can be larger than this
  int nbins_;

  int nb1_;

  std::vector<StatType> bins1_;
  std::vector<StatType> bins2_;

  int last1_;
  int last2_;

  int nproc_;

public:
  Binning(int nbins = 16) { Init(nbins); }

  void Init(int nbins) {
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_);

    nbins_ = nbins;

    if (nbins_ % 2 != 0) {
      std::cerr << "#The number of bins should be a multiple of two"
                << std::endl;
    }

    bins1_.resize(nbins);
    bins2_.resize(nbins);

    Reset();
  }

  void Reset() {
    last1_ = 0;
    last2_ = 0;
    nb1_ = 0;

    for (std::size_t i = 0; i < bins1_.size(); i++) {
      bins1_[i].Reset();
    }

    for (std::size_t i = 0; i < bins2_.size(); i++) {
      bins2_[i].Reset();
    }
  }

  void operator<<(const DataType &val) {
    if (last1_ < nbins_) {
      bins1_[last1_] << val;
      last1_++;
    } else {
      nb1_ = bins1_[0].N();

      if (last2_ < nbins_) {
        bins2_[last2_] << val;
        if (bins2_[last2_].N() == nb1_) {
          last2_++;
        }
      } else {
        Merge();
      }
    }
  }

  void Print() {
    for (std::size_t i = 0; i < last1_; i++) {
      std::cout << bins1_[i].Mean() << "  " << bins1_[i].N() << std::endl;
    }
    for (std::size_t i = 0; i < last2_; i++) {
      std::cout << bins2_[i].Mean() << "  " << bins2_[i].N() << std::endl;
    }
  }

  void Merge() {

    for (std::size_t i = 0; i < bins1_.size(); i += 2) {
      bins1_[i] << bins1_[i + 1];
      bins2_[i] << bins2_[i + 1];
    }
    for (std::size_t i = 0; i < bins1_.size() / 2; i++) {
      bins1_[i] = bins1_[2 * i];
    }
    for (std::size_t i = 0; i < bins1_.size() / 2; i++) {
      bins1_[i + bins1_.size() / 2] = bins2_[2 * i];
    }

    last2_ = 0;
    for (std::size_t i = 0; i < bins2_.size(); i++) {
      bins2_[i].Reset();
    }
  }

  // Returns binned statistics
  StatType Binned() const {
    StatType bint;

#ifndef NDEBUG
    int nb = bins1_[0].N();
#endif
    for (int i = 0; i < last1_; i++) {
      bint << bins1_[i].Mean();
      assert(bins1_[i].N() == nb);
    }
    for (int i = 0; i < last2_; i++) {
      bint << bins2_[i].Mean();
      assert(bins2_[i].N() == nb);
    }
    return bint;
  }

  // Returns unbinned statistics
  StatType UnBinned() const {
    StatType bint;

    for (int i = 0; i < last1_; i++) {
      bint << bins1_[i];
    }
    for (int i = 0; i < last2_; i++) {
      bint << bins2_[i];
    }
    return bint;
  }

  // Estimates the auto-correlation time using error of mean
  // computed on blocked values
  DataType TauCorrProc() const {
    const auto unbinned = UnBinned();
    const auto binned = Binned();

    const auto erbinned = binned.ErrorOfMean();
    const auto erunbinned = unbinned.ErrorOfMean();

    return TauCorrOp(erbinned, erunbinned);
  }

  DataType TauCorr() const {
    DataType taucp = TauCorrProc();

    SumOnNodes(taucp);

    return taucp / double(nproc_);
  }

  DataType Mean() {
    const auto binned = Binned();

    DataType mean = binned.Mean();

    SumOnNodes(mean);

    return mean / double(nproc_);
  }

  DataType ErrorOfMean() {
    const auto binned = Binned();

    DataType sigma2 = Sigma2Op(binned.ErrorOfMean());
    SumOnNodes(sigma2);

    return EoMeanOp(sigma2);
  }

  json AllStats() const {
    const auto binned = Binned();

    DataType mean = binned.Mean();
    SumOnNodes(mean);
    mean /= double(nproc_);

    DataType sigma2 = Sigma2Op(binned.ErrorOfMean());
    SumOnNodes(sigma2);
    auto eomean = EoMeanOp(sigma2);

    auto tcorr = TauCorr();

    json j;
    j["Mean"] = mean;
    j["Sigma"] = eomean;
    j["Taucorr"] = tcorr;

    return j;
  }

  int NvalProc() {
    int nv = 0;

    for (int i = 0; i < last1_; i++) {
      nv += bins1_[i].N();
    }
    for (int i = 0; i < last2_; i++) {
      nv += bins2_[i].N();
    }
    return nv;
  }

  int N() {
    int Np = NvalProc();
    SumOnNodes(Np);
    return Np;
  }

  double TauCorrOp(double erbinned, double erunbinned) const {
    return 0.5 * (std::pow(erbinned / erunbinned, 2.) - 1.);
  }

  Eigen::VectorXd TauCorrOp(const Eigen::VectorXd &erbinned,
                            const Eigen::VectorXd &erunbinned) const {
    Eigen::VectorXd result(erbinned.size());

    for (int i = 0; i < result.size(); i++) {
      result(i) = 0.5 * (std::pow(erbinned(i) / erunbinned(i), 2.) - 1.);
    }
    return result;
  }

  template <class P>
  std::vector<P> TauCorrOp(const std::vector<P> &erbinned,
                           const std::vector<P> &erunbinned) const {
    std::vector<P> result(erbinned.size());

    for (int i = 0; i < result.size(); i++) {
      result[i] = 0.5 * (std::pow(erbinned[i] / erunbinned[i], 2.) - 1.);
    }
    return result;
  }

  template <class P>
  std::valarray<P> TauCorrOp(const std::valarray<P> &erbinned,
                             const std::valarray<P> &erunbinned) const {
    return 0.5 * (std::pow(erbinned / erunbinned, 2.) - 1.);
  }

  double EoMeanOp(double sigma2) const {
    return std::sqrt(sigma2) / (double(nproc_));
  }

  Eigen::VectorXd EoMeanOp(const Eigen::VectorXd &sigma2) const {
    Eigen::VectorXd result(sigma2.size());

    for (int i = 0; i < result.size(); i++) {
      result(i) = EoMeanOp(sigma2(i));
    }
    return result;
  }

  template <class P>
  std::vector<P> EoMeanOp(const std::vector<P> &sigma2) const {
    std::vector<P> result(sigma2.size());

    for (int i = 0; i < result.size(); i++) {
      result[i] = EoMeanOp(sigma2(i));
    }
    return result;
  }

  template <class P>
  std::valarray<P> EoMeanOp(const std::valarray<P> &sigma2) const {
    return std::sqrt(sigma2) / (double(nproc_));
  }

  double Sigma2Op(double sigma2) const { return sigma2 * sigma2; }

  Eigen::VectorXd Sigma2Op(const Eigen::VectorXd &sigma2) const {
    Eigen::VectorXd result(sigma2.size());

    for (int i = 0; i < result.size(); i++) {
      result(i) = Sigma2Op(sigma2(i));
    }
    return result;
  }

  template <class P>
  std::vector<P> Sigma2Op(const std::vector<P> &sigma2) const {
    std::vector<P> result(sigma2.size());

    for (int i = 0; i < result.size(); i++) {
      result[i] = Sigma2Op(sigma2(i));
    }
    return result;
  }

  template <class P>
  std::valarray<P> Sigma2Op(const std::valarray<P> &sigma2) const {
    return sigma2 * sigma2;
  }
};

} // namespace netket

#endif
