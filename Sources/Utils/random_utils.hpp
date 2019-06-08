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

#ifndef NETKET_RANDOMUTILS_HPP
#define NETKET_RANDOMUTILS_HPP

#include <complex>
#include <random>

#include <mpi.h>
#include <Eigen/Dense>

#include "Utils/mpi_interface.hpp"
#include "common_types.hpp"

namespace netket {
using default_random_engine = std::mt19937;

inline void RandomGaussian(Eigen::Matrix<double, Eigen::Dynamic, 1> &par,
                           int seed, double sigma) {
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0, sigma);
  for (int i = 0; i < par.size(); i++) {
    par(i) = distribution(generator);
  }
}

inline void RandomGaussian(Eigen::Matrix<Complex, Eigen::Dynamic, 1> &par,
                           int seed, double sigma) {
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0, sigma);
  for (int i = 0; i < par.size(); i++) {
    par(i) = Complex(distribution(generator), distribution(generator));
  }
}

/**
 * Random engine that supports deterministic seeding but uses a different
 * derived seed for every MPI process.
 */
class DistributedRandomEngine {
 public:
  using ResultType = default_random_engine::result_type;

  /**
   * Construct the engines with a non-deterministic base seed (obtained from
   * std::random_device).
   */
  DistributedRandomEngine() : engine_(GetDerivedSeed()) {}

  /**
   * Construct the engines with the given base_seed.
   */
  explicit DistributedRandomEngine(ResultType base_seed)
      : engine_(GetDerivedSeed(base_seed)) {}

  /**
   * Returns the underlying random engine.
   */
  default_random_engine &Get() { return engine_; }

  /**
   * Resets the seeds of the random engines of all MPI processes from the given
   * base_seed.
   */
  void Seed(ResultType base_seed) { engine_.seed(GetDerivedSeed(base_seed)); }

 private:
  default_random_engine engine_;

  /**
   * Generate seeds for all MPI processes pseudo-randomly from a
   * non-deterministic base seed (which is obtained from std::random_device).
   */
  ResultType GetDerivedSeed() {
    std::random_device rd;
    return GetDerivedSeed(rd());
  }

  /**
   * Generate seeds for all MPI processes pseudo-randomly from the base seed.
   * @param baseseed Seed used to initialize the RNG which generates the seeds
   *    of the RNGs for all MPI processes.
   */
  ResultType GetDerivedSeed(ResultType base_seed) {
    int rank_s;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_s);
    int size_s;
    MPI_Comm_size(MPI_COMM_WORLD, &size_s);
    const auto rank = static_cast<size_t>(rank_s);
    const auto size = static_cast<size_t>(size_s);

    std::vector<ResultType> seeds;
    seeds.resize(size);

    default_random_engine seed_engine(base_seed);
    std::uniform_int_distribution<ResultType> dist;

    if (rank == 0) {
      for (std::size_t i = 0; i < size; ++i) {
        seeds[i] = dist(seed_engine);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    SendToAll(seeds);

    return seeds[rank];
  }
};

}  // namespace netket

#endif
