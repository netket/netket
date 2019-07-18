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
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

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

  explicit SR(double diagshift = 0.01, bool use_iterative = false,
              bool use_cholesky = true, bool is_holomorphic = true)
      : sr_diag_shift_(diagshift),
        use_iterative_(use_iterative),
        use_cholesky_(use_cholesky),
        is_holomorphic_(is_holomorphic) {
    InfoMessage() << GetInfoString();
  }

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

  void SetParameters(double diagshift = 0.01, bool use_iterative = false,
                     bool use_cholesky = true, bool is_holomorphic = true) {
    sr_diag_shift_ = diagshift;
    use_iterative_ = use_iterative;
    use_cholesky_ = use_cholesky;
    is_holomorphic_ = is_holomorphic;
  }

  std::string GetInfoString() {
    std::stringstream str;
    str << "Using the Stochastic reconfiguration method for "
        << (is_holomorphic_ ? "holomorphic" : "real-parameter")
        << "wavefunctions\n";
    if (use_iterative_) {
      str << "With iterative solver";
    } else {
      if (use_cholesky_) {
        str << "Using Cholesky decomposition";
      } else {
        str << "Using BDCSVD decomposition";
      }
    }
    return str.str();
  }

 private:
  double sr_diag_shift_;
  bool use_iterative_;
  bool use_cholesky_;
  bool is_holomorphic_;

  Eigen::MatrixXd Sreal_;
  Eigen::MatrixXcd Scomplex_;

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
    if (use_cholesky_) {
      Eigen::LLT<Mat> llt(A);
      deltaP = llt.solve(b);
    } else {
      const constexpr auto options = Eigen::ComputeThinU | Eigen::ComputeThinV;
      Eigen::BDCSVD<Mat> bdcsvd(A, options);
      deltaP = bdcsvd.solve(b);
    }
  }
};

}  // namespace netket

#endif
