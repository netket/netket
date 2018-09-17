// Copyright 2018 Alexander Wietek - All Rights Reserved.
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

#ifndef NETKET_EXACT_DIAGONALIZATION_HPP
#define NETKET_EXACT_DIAGONALIZATION_HPP

#include <vector>

#include <ietl/lanczos.h>
#include <ietl/randomgenerator.h>

#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"

namespace netket {
  
  namespace detail {
    template <class matrix_t, class iter_t, class random_t>
    std::vector<double>eigenvalues_lanczos_run(const matrix_t& matrix,
					       const random_t& random_gen,
					       iter_t& iter) {
      using complex = std::complex<double>;
      using vectorspace_t = ietl::vectorspace<complex>;
      using lanczos_t = ietl::lanczos<matrix_t, vectorspace_t>;
      size_t dimension = matrix.GetDimension();
      vectorspace_t ietl_vecspace(dimension);
      lanczos_t lanczos(matrix, ietl_vecspace);
      lanczos.calculate_eigenvalues(iter, random_gen);
      if (iter.error_code() == 1)
	printf("Warning: Lanczos algorithm did NOT converge in %i steps!\n",
	       iter.max_iterations());      
      return lanczos.eigenvalues();
    }
  }
  
  std::vector<double> eigenvalues_lanczos(const Hamiltonian &hamiltonian,
					  bool matrix_free=false,
					  int first_n = 1,
					  int max_iter = 1000, int seed=42,
					  double precision = 1e-14) {
    using normal_dist_t = std::uniform_real_distribution<double>;
    using random_t = ietl::random_generator<std::mt19937, normal_dist_t>;
    using iter_t = ietl::lanczos_iteration_nlowest<double>;

    normal_dist_t dist(-1., 1.);
    random_t random_gen(dist, seed);

    // Converge the first_n eigenvalues to precision
    iter_t iter(max_iter, first_n, sqrt(precision), sqrt(precision));

    std::vector<double> eigs_lanczos;
  
    if (matrix_free) { // Matrix-free computation
      using matrix_t = DirectMatrixWrapper<Hamiltonian>;
      matrix_t matrix(hamiltonian);
      eigs_lanczos = detail::eigenvalues_lanczos_run(matrix, random_gen,
						     iter);
    
    } else { // computation using Sparse matrix
      using matrix_t = SparseMatrixWrapper<Hamiltonian>;
      matrix_t matrix(hamiltonian);
      eigs_lanczos = detail::eigenvalues_lanczos_run(matrix, random_gen,
						     iter);
    }
    eigs_lanczos.resize(first_n); // Keep only converged eigenvalues
    return eigs_lanczos;
  }

  std::vector<double> eigenvalues_full(const Hamiltonian &hamiltonian,
				       int first_n = 1) {
    SparseMatrixWrapper<Hamiltonian> matrix(hamiltonian);
    auto ed = matrix.ComputeEigendecomposition(Eigen::EigenvaluesOnly);
    auto eigs = ed.eigenvalues();
    eigs.conservativeResize(first_n); // Keep only first_n eigenvalues

    return std::vector<double>(eigs.data(),
			       eigs.data() + eigs.rows() * eigs.cols());
  }

  void write_ed_eigenvalues(const json &pars,
			    const std::vector<double>& eigs)
  {
    std::string file_base = FieldVal(pars["GroundState"], "OutputFile");
    std::string file_name = file_base + std::string(".log");
    std::ofstream file_ed(file_name);
    json j(eigs);
    file_ed << j << std::endl;
    file_ed.close();
  }

  void get_ed_parameters(const json &pars, double& precision,
			 int& n_eigenvalues, int& random_seed, int& max_iter)
  {
    n_eigenvalues = FieldExists(pars["GroundState"], "NumEigenvalues") ?
      static_cast<int>(FieldVal(pars["GroundState"], "NumEigenvalues")) :
      1;
    precision = FieldExists(pars["GroundState"], "Precision") ?
      static_cast<double>(FieldVal(pars["GroundState"], "Precision")) :
      1e-14;
    random_seed = FieldExists(pars["GroundState"], "RandomSeed") ?
      static_cast<int>(FieldVal(pars["GroundState"], "RandomSeed")) :
      42;
    max_iter = FieldExists(pars["GroundState"], "MaxIterations") ?
      static_cast<int>(FieldVal(pars["GroundState"], "MaxIterations")) :
      1000;
  }

}  // namespace netket

#endif
