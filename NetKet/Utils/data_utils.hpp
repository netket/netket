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

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Utils/all_utils.hpp"

#ifndef NETKET_UNIFORM_DATA
#define NETKET_UNIFORM_DATA

namespace netket {

/**
  Represents data (x, phi(x))
*/
template <typename T>
class Data {
  // Memory for the amplitudes and configs can be allocated once
  // we read the Hilbert space info from the json file. Might be
  // more efficient that way!
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  // amplitudes with dimension (ndata_, 2)
  // since amp is in general complex
  MatrixType amplitudes;

  // basis with dimension (ndata_, nv_)
  MatrixType basis;

  /**
    Hilbert space for describing the space of the input data
  */
  Hilbert hilbert_;

  /// Number of data points
  unsigned int ndata_;

 public:
  /// Constructor
  explicit Data(const json &pars) { ReadFromJson(pars); }

  /// Accessor function
  unsigned int GetNdata() const { return ndata_; }
  /// Accessor function
  const Hilbert &GetHilbert() const override { return hilbert_; }

  /**
    Reads data from a Json object
  */
  void ReadFromJson(const json &pars) {
    // Extract and initialize the Hilbert space
    hilbert_.Init(pars);

    // Given the info Nsites from the Hilbert space, we can
    // allocate memory for the configurations and the amplitudes
    nv_ = pars["Hilbert"]["Nsites"];

    // Extract labels
    amplitudes = pars["samples"]["amp"];
    // Extract inputs
    configs = pars["samples"]["basis"];

    // Extract input and label properties
    ndata_ = amplitudes.rows();

    /// \todo Should build in a check for this?
    // nv_         = amplitudes.cols();

    // Debug info
    std::cout << " read in amplitudes as array \n"
              << amplitudes << "\n"
              << "read in basis as array \n"
              << basis << "\n";
  }

  void GenerateBatch(unsigned int batchsize, MatrixType &out) {
    // Clip batchsize to number of samples
    if (batchsize >= ndata_) batchsize = ndata_;

    /// \todo Implement shuffling
    /// \todo Randomly pick 'batchsize' samples and store in out
  }
};

}  // namespace netket

#endif
