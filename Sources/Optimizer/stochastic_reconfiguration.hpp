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

/**
 * Stochastic reconfiguration solver.
 */
class SR {
 public:
  using OkRef = Eigen::Ref<const RowMatrixXcd>;
  using GradRef = Eigen::Ref<const VectorXcd>;
  using OutputRef = Eigen::Ref<VectorXcd>;

  enum LSQSolver { LLT = 0, LDLT = 1, ColPivHouseholder = 2, BDCSVD = 3 };

  static nonstd::optional<LSQSolver> SolverFromString(const std::string& name);
  static const char* SolverAsString(LSQSolver solver);

  explicit SR(LSQSolver solver, double diagshift = 0.01,
              bool use_iterative = false, bool is_holomorphic = true)
      : solver_(solver),
        sr_diag_shift_(diagshift),
        use_iterative_(use_iterative),
        is_holomorphic_(is_holomorphic) {}

  explicit SR(double diagshift = 0.01, bool use_iterative = false,
              bool use_cholesky = true, bool is_holomorphic = true)
      __attribute__((deprecated))
      : SR(use_cholesky ? LLT : ColPivHouseholder, diagshift, use_iterative,
           is_holomorphic) {}

  /**
   * Solves the SR flow equation for the parameter update ẋ.
   *
   * The SR update is computed by solving the linear euqation
   *    Sẋ = f
   * where S is the covariance matrix of the partial derivatives
   * O_i(v_j) = ∂/∂x_i log Ψ(v_j) and f is a generalized force (the loss
   * gradient).
   *
   * @param Oks The matrix 𝕆 of centered log-derivatives,
   *    𝕆_ij = O_i(v_j) - ⟨O_i⟩.
   * @param grad The loss gradient f.
   * @param deltaP Output parameter for the update ẋ.
   */
  void ComputeUpdate(OkRef Oks, GradRef grad, OutputRef deltaP);

  void SetParameters(LSQSolver solver, double diagshift = 0.01,
                     bool use_iterative = false, bool is_holomorphic = true);
  void SetParameters(double diagshift = 0.01, bool use_iterative = false,
                     bool use_cholesky = true, bool is_holomorphic = true)
      __attribute__((deprecated));

  /**
   * Returns a string describing the current parameters of the SR class.
   */
  std::string LongDesc(Index depth = 0) const;
  std::string ShortDesc() const;

  /**
   * Becca and Sorella (2017), pp. 143-144.
   */
  void SetScaleInvariantRegularization(bool enabled) {
    if (use_iterative_ && enabled) {
      // TODO: implement
      throw std::runtime_error{
          "Scale-invariant regularization is not implemented "
          "for iterative solvers at the moment."};
    }
    if (enabled) {
      InfoMessage() << "Using scale-invariant preconditioning." << std::endl;
    }
    scale_invariant_pc_ = enabled;
  }
  bool ScaleInvariantRegularizationEnabled() const {
    return scale_invariant_pc_;
  }

  /**
   * Returns the rank of the S matrix computed during the last call to
   * `ComputeUpdate` or `nullopt`, in case storing the rank is not enabled
   * and before the first call to `ComputeUpdate`.
   *
   * Storing the rank is enabled and disabled by `SetStoreRank` below.
   */
  nonstd::optional<Index> LastRank() { return last_rank_; }
  bool StoreRankEnabled() const { return store_rank_; }
  void SetStoreRank(bool enabled);

  /**
   * Returns the S matrix computed during the last call to
   * `ComputeUpdate` or `nullopt`, in case storing the S matrix is not enabled
   * and before the first call to `ComputeUpdate`.
   *
   * Storing the S matrix is enabled and disabled by `SetStoreFullSMatrix`
   * below.
   */
  const nonstd::optional<MatrixXcd>& LastSMatrix() const {
    return last_S_;
  }
  bool StoreFullSMatrixEnabled() const { return store_full_S_matrix_; }
  void SetStoreFullSMatrix(bool enabled);

 private:
  LSQSolver solver_;
  double sr_diag_shift_;
  bool use_iterative_;
  bool is_holomorphic_;

  bool scale_invariant_pc_ = false;
  VectorXd diag_S_;

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
  }

  template <class Mat, class Vec>
  void ApplyPreconditioning(Mat& S, Vec& grad) {
    if (scale_invariant_pc_) {
      // Even if S is complex, its diagonal elements should be real since it
      // is Hermitian.
      diag_S_ = S.diagonal().real().cwiseSqrt();

      static const double CUTOFF = 1e-10;
      for (Index i = 0; i < diag_S_.rows(); i++) {
        if (diag_S_(i) <= CUTOFF) {
          diag_S_(i) = 1.0;
          S.col(i).setZero();
          S.row(i).setZero();
          S(i, i) = 1.0;
        }
      }

      S.array() /= (diag_S_ * diag_S_.transpose()).array();
      grad.array() /= diag_S_.array();
    }
    // Apply diagonal shift
    S.diagonal().array() += sr_diag_shift_;
  }

  void RevertPreconditioning(OutputRef solution) {
    if (scale_invariant_pc_) {
      solution.array() /= diag_S_.array();
    }
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
    SolverType it_solver(S);
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
    } else if (solver_ == LDLT) {
      Eigen::LDLT<Mat> ldlt(A);
      deltaP = ldlt.solve(b);
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
    if (solver == LLT || solver == LDLT) {
      std::stringstream str;
      str << "SR cannot store matrix rank: Solver " << SolverAsString(solver)
          << " is not rank-revealing.";
      throw std::logic_error{str.str()};
    }
  }
};

}  // namespace netket

#endif
