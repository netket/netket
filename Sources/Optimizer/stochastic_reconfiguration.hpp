// Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_SR_HPP
#define NETKET_SR_HPP

#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <nonstd/optional.hpp>

#include "Utils/messages.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "common_types.hpp"
#include "matrix_replacement.hpp"

namespace netket {

// Generalized Stochastic Reconfiguration Updates
class SR {
 public:
  using OkRef = Eigen::Ref<const Eigen::MatrixXcd>;
  using GradRef = Eigen::Ref<const Eigen::VectorXcd>;
  using OutputRef = Eigen::Ref<Eigen::VectorXcd>;

  enum LSQSolver { LLT = 0, ColPivHouseholder = 1, BDCSVD = 2 };

  static nonstd::optional<LSQSolver> SolverFromString(const std::string& name) {
    if (name == "LLT") {
      return LLT;
    } else if (name == "ColPivHouseholder") {
      return ColPivHouseholder;
    } else if (name == "BDCSVD") {
      return BDCSVD;
    } else {
      return nonstd::nullopt;
    }
  }

  static const char* SolverAsString(LSQSolver solver) {
    static const char* solvers[] = {"LLT", "ColPivHouseholder", "BCDSVD"};
    return solvers[solver];
  }

  explicit SR(LSQSolver solver, double diagshift = 0.01,
              bool use_iterative = false, bool is_holomorphic = true)
      : solver_(solver),
        sr_diag_shift_(diagshift),
        use_iterative_(use_iterative),
        is_holomorphic_(is_holomorphic) {
    InfoMessage() << GetInfoString();
  }

  explicit SR(double diagshift = 0.01, bool use_iterative = false,
              bool use_cholesky = true, bool is_holomorphic = true)
      __attribute__((deprecated))
      : SR(use_cholesky ? LLT : ColPivHouseholder, diagshift, use_iterative,
           is_holomorphic) {}

  void ComputeUpdate(OkRef Oks, GradRef grad, OutputRef deltaP) {
    double nsamp = Oks.rows();
    SumOnNodes(nsamp);
    // auto npar = grad.size();

    if (is_holomorphic_) {
      if (use_iterative_) {
        SolveIterative<VectorXcd>(Oks, grad, deltaP, nsamp);
      } else {
        BuildSMatrix<MatrixXcd>(Oks.adjoint() * Oks, Scomplex_, nsamp);
        SolveLeastSquares<MatrixXcd, VectorXcd>(Scomplex_, grad, deltaP);
      }
    } else {
      if (use_iterative_) {
        SolveIterative<VectorXd>(Oks, grad.real(), deltaP, nsamp);
      } else {
        BuildSMatrix<MatrixXd>((Oks.adjoint() * Oks).real(), Sreal_, nsamp);
        SolveLeastSquares<MatrixXd, VectorXd>(Sreal_, grad.real(),
                                              deltaP.real());
      }
      deltaP.imag().setZero();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void SetParameters(LSQSolver solver, double diagshift = 0.01,
                     bool use_iterative = false, bool is_holomorphic = true) {
    CheckSolverCompatibility(use_iterative, solver, store_rank_);

    solver_ = solver;
    sr_diag_shift_ = diagshift;
    use_iterative_ = use_iterative;
    is_holomorphic_ = is_holomorphic;
  }

  void SetParameters(double diagshift = 0.01, bool use_iterative = false,
                     bool use_cholesky = true, bool is_holomorphic = true)
      __attribute__((deprecated)) {
    SetParameters(use_cholesky ? LLT : ColPivHouseholder, diagshift,
                  use_iterative, is_holomorphic);
  }

  std::string GetInfoString() {
    std::stringstream str;
    str << "Using the Stochastic reconfiguration method for "
        << (is_holomorphic_ ? "holomorphic" : "real-parameter")
        << " wavefunctions\n";
    if (use_iterative_) {
      str << "With iterative solver";
    } else {
      str << "Using " << SolverAsString(solver_) << " solver";
    }
    str << "\n";
    return str.str();
  }

  bool StoreRankEnabled() const { return store_rank_; }
  void SetStoreRank(bool enabled) {
    CheckSolverCompatibility(use_iterative_, solver_, enabled);
    store_rank_ = enabled;
    if (!enabled) {
      last_rank_ = nonstd::nullopt;
    }
  }
  /**
   * Returns the rank of the S matrix computed during the last call to
   * `ComputeUpdate` or `nullopt`, in case storing the rank is not enabled
   * and before the first call to `ComputeUpdate`.
   */
  nonstd::optional<Index> LastRank() { return last_rank_; }

  bool StoreFullSMatrixEnabled() const { return store_full_S_matrix_; }
  void SetStoreFullSMatrix(bool enabled) {
    if (use_iterative_ && enabled) {
      throw std::logic_error{
          "Cannot store full S matrix with `use_iterative = true`."};
    }
    store_full_S_matrix_ = enabled;
    if (!enabled) {
      last_S_ = nonstd::nullopt;
    }
  }
  nonstd::optional<MatrixXcd> LastSMatrix() const { return last_S_; }

 private:
  LSQSolver solver_;
  double sr_diag_shift_;
  bool use_iterative_;
  bool is_holomorphic_;

  Eigen::MatrixXd Sreal_;
  Eigen::MatrixXcd Scomplex_;

  bool store_rank_ = false;
  bool store_full_S_matrix_ = false;

  nonstd::optional<Index> last_rank_;
  nonstd::optional<MatrixXcd> last_S_;

  template <class Mat>
  void BuildSMatrix(Eigen::Ref<const Mat> S_local, Mat& S_out, double nsamp) {
    static_assert(std::is_same<Mat, MatrixXd>::value ||
                      std::is_same<Mat, MatrixXcd>::value,
                  "S_out must be of type MatrixXd or MatrixXcd");

    S_out = S_local;
    SumOnNodes(S_out);
    S_out /= nsamp;

    S_out.diagonal().array() += sr_diag_shift_;
  }

  template <class Vec>
  void SolveIterative(OkRef Oks, Eigen::Ref<const Vec> grad, OutputRef deltaP,
                      double nsamp) {
    // Use SrMatrixReal iff Vec is a real vector, else SrMatrixComplex
    using IsReal = std::is_same<typename Vec::Scalar, double>;
    using SrMatrixType = typename std::conditional<IsReal::value, SrMatrixReal,
                                                   SrMatrixComplex>::type;
    // Make sure this code does not compile by accident if grad is a different
    // type
    static_assert(std::is_same<typename Vec::Scalar, double>::value ||
                      std::is_same<typename Vec::Scalar, Complex>::value,
                  "grad must be a real or complex vector");

    SrMatrixType S;
    S.attachMatrix(Oks);
    S.setShift(sr_diag_shift_);
    S.setScale(1. / nsamp);

    using SolverType =
        Eigen::ConjugateGradient<SrMatrixType, Eigen::Lower | Eigen::Upper,
                                 Eigen::IdentityPreconditioner>;
    SolverType it_solver;
    it_solver.setTolerance(1.0e-3);
    deltaP = it_solver.solve(grad);
  }

  template <class Mat, class Vec, class Out>
  void SolveLeastSquares(Mat& A, Eigen::Ref<const Vec> b, Out&& deltaP) {
    if (store_full_S_matrix_) {
      last_S_.emplace(A);
    }

    if (solver_ == LLT) {
      Eigen::LLT<Mat> llt(A);
      deltaP = llt.solve(b);
    } else if (solver_ == ColPivHouseholder) {
      Eigen::ColPivHouseholderQR<Mat> qr(A);
      deltaP = qr.solve(A);
      if (store_rank_) {
        last_rank_ = qr.rank();
      }
    } else if (solver_ == BDCSVD) {
      const constexpr auto options = Eigen::ComputeThinU | Eigen::ComputeThinV;
      Eigen::BDCSVD<Mat> bdcsvd(A, options);
      deltaP = bdcsvd.solve(b);
      if (store_rank_) {
        last_rank_ = bdcsvd.rank();
      }
    } else {
      throw std::runtime_error{
          "Unknown LSQSolver enum value in SR. This should never happen."};
    }
  }

  static void CheckSolverCompatibility(bool use_iterative, LSQSolver solver,
                                       bool rank_enabled) {
    if (!rank_enabled) {
      return;
    }
    if (use_iterative) {
      throw std::logic_error{
          "SR cannot store matrix rank with interactive solver."};
    }
    if (solver == LLT) {
      std::stringstream str;
      str << "SR cannot store matrix rank: Solver " << SolverAsString(solver)
          << " is not rank-revealing.";
      throw std::logic_error{str.str()};
    }
  }
};

}  // namespace netket

#endif
