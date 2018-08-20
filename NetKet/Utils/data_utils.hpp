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
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <vector>
#include "Utils/all_utils.hpp"
#include <complex>


#ifndef NETKET_UNIFORM_DATA
#define NETKET_UNIFORM_DATA

namespace netket {

// Hash function for Eigen matrix and vector.
// The code is from `hash_combine` function of the Boost library. See
// http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
	  // Note that it is oblivious to the storage order of Eigen matrix (column- or
	  // row-major). It will give you the same hash value for two different matrices if they
	  // are the transpose of each other in different storage order.
  size_t seed = 0;
  for (size_t i = 0; i < matrix.size(); ++i) {
	  auto elem = *(matrix.data() + i);
	  seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
  }
};


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

  // configs with dimension (ndata_, nv_)
  MatrixType configs;

  /**
    Hilbert space for describing the space of the input data
  */
  Hilbert hilbert_;

  /// Number of data points
  unsigned int ndata_;
  /// Number of system size, i.e. visible units
  unsigned int nv_;

  /// map from config vector to amp
  std::unordered_map<VectorType, std::complex<double>, matrix_hash<VectorType>> config_2_amp;

 public:
  /// Constructor
  explicit Data(const json &pars, const json &supervisedPars) { ReadFromJson(pars, supervisedPars); }

  /// Accessor function
  unsigned int GetNdata() const { return ndata_; }
  /// Accessor function
  const Hilbert& GetHilbert() { return hilbert_; }

  /**
    Reads data from a Json object
  */
  void ReadFromJson(const json &pars, const json &supervisedPars) {
    // Extract and initialize the Hilbert space
    hilbert_.Init(supervisedPars);

    // Given the info Size from the Hilbert space, we can
    // allocate memory for the configurations and the amplitudes
    nv_ = hilbert_.Size();

    // Extract labels
    amplitudes = pars["samples"]["amp"];
    // Extract inputs
    configs = pars["samples"]["configs"];

    // Extract input and label properties
    ndata_ = amplitudes.rows();

    /// \todo Should build in a check for this?
    // nv_         = amplitudes.cols();

    // Debug info
    std::cout << " read in amplitudes as array \n"
              << amplitudes << "\n"
              << "read in configs as array \n"
              << configs << "\n";

    std::cout << " construct mapping between configs and amplitudes \n";
    for(int i=0; i<configs.rows(); i++){
        std::complex<double> amp_complex(amplitudes(i,0),amplitudes(i,1));
        std::cout << "mapping : " << configs.row(i)
                  << " to " << amp_complex <<"\n";
        config_2_amp[configs.row(i)] = amp_complex;
    }
  }

  void GenerateBatch(unsigned int batchsize, MatrixType &out) {
    // Clip batchsize to number of samples
    if (batchsize >= ndata_) batchsize = ndata_;

    /// \todo Implement shuffling
    /// \todo Randomly pick 'batchsize' samples and store in out
  }

  std::complex<double> logVal(VectorType v){
    return std::log(config_2_amp[v]);
  }

  std::complex<double> Val(VectorType v){
    return config_2_amp[v];
  }


};

}  // namespace netket

#endif
