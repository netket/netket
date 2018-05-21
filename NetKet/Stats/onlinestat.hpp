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

#ifndef NETKET_ONLINESTAT_HPP
#define NETKET_ONLINESTAT_HPP

#include <mpi.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <vector>
#include "Utils/random_utils.hpp"

namespace netket {

/// Online statistics
// for general types
// this class accumulates results
// with minimal memory requirements
// simple statistics can then be obtained
// or merged with other bins

template <class T>
class OnlineStat {
  // Number of samples in this bin
  int N_;

  // current mean
  T mean_;

  // Running sum of squares of differences from the current mean
  T m2_;

  bool firstcall_;

 public:
  using DataType = T;

  explicit OnlineStat() { Reset(); }

  // Adding data to this bin
  inline void operator<<(const DataType &data) {
    CheckCall(data);

    N_ += 1;
    const T delta = data - mean_;
    mean_ += delta / double(N_);

    const T delta2 = data - mean_;
    m2_ += delta * delta2;
  }

  // Merging with another bin
  inline void operator<<(const OnlineStat<T> &obin) {
    CheckCall(obin.Mean());

    N_ += obin.N();
    const T delta = obin.Mean() - Mean();
    mean_ += delta * obin.N() / double(N_);

    m2_ += obin.m2_ + delta * delta * obin.N() * (1. - obin.N() / double(N_));
  }

  inline int N() const { return N_; }

  inline DataType Mean() const { return mean_; }

  inline DataType Variance() const { return m2_ / double(N_ - 1.); }

  inline DataType ErrorOfMean() const {
    return sqrt(m2_ / double(N_ * (N_ - 1.)));
  }

  void Reset() { firstcall_ = true; }

  void CheckCall(const DataType &data) {
    if (firstcall_) {
      N_ = 0;
      mean_ = DataType(data);
      mean_ = 0;
      m2_ = DataType(data);
      m2_ = 0;
      firstcall_ = false;
    }
  }
};

/// Online statistics
// for scalars
// this class accumulates results
// with minimal memory requirements
// simple statistics can then be obtained
// or merged with other bins
template <>
class OnlineStat<Eigen::VectorXd> {
  // Number of samples in this bin
  int N_;

  // current mean
  Eigen::VectorXd mean_;

  // Running sum of squares of differences from the current mean
  Eigen::VectorXd m2_;

  bool firstcall_;

 public:
  using DataType = Eigen::VectorXd;

  explicit OnlineStat() { Reset(); }

  // Adding data to this bin
  inline void operator<<(const DataType &data) {
    CheckCall(data);

    N_ += 1;
    const DataType delta = data - mean_;
    mean_ += delta / double(N_);

    const DataType delta2 = data - mean_;
    m2_ += delta.cwiseProduct(delta2);
  }

  // Merging with another bin
  inline void operator<<(const OnlineStat<Eigen::VectorXd> &obin) {
    CheckCall(obin.Mean());

    N_ += obin.N();
    const DataType delta = obin.Mean() - Mean();
    mean_ += delta * obin.N() / double(N_);
    m2_ += obin.m2_ +
           delta.cwiseProduct(delta) * obin.N() * (1. - obin.N() / double(N_));
  }

  inline int N() const { return N_; }

  inline DataType Mean() const { return mean_; }

  inline DataType Variance() const { return m2_ / double(N_ - 1.); }

  inline DataType ErrorOfMean() const {
    return m2_.cwiseSqrt() / std::sqrt(double(N_ * (N_ - 1.)));
  }

  void Reset() { firstcall_ = true; }

  void CheckCall(const DataType &data) {
    if (firstcall_) {
      N_ = 0;
      mean_.resize(data.size());
      mean_.setZero();
      m2_.resize(data.size());
      m2_.setZero();
      firstcall_ = false;
    }
  }
};

}  // namespace netket
#endif
