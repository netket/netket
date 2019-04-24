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
  double sr_diag_shift_;
  bool use_iterative_;
  bool use_cholesky_;
  bool is_holomorphic_;

  Eigen::MatrixXd Sreal_;
  Eigen::MatrixXcd Scomplex_;

 public:
  SR() { setDefaultParameters(); }

  void ComputeUpdate(const Eigen::Ref<const Eigen::MatrixXcd> Oks,
                     const Eigen::Ref<const Eigen::VectorXcd> grad,
                     Eigen::Ref<Eigen::VectorXcd> deltaP) {
    double nsamp = Oks.rows();

    SumOnNodes(nsamp);
    auto npar = grad.size();
    if (is_holomorphic_) {
      if (!use_iterative_) {
        // Explicit construction of the S matrix
        Scomplex_.resize(npar, npar);
        Scomplex_ = (Oks.adjoint() * Oks);
        SumOnNodes(Scomplex_);
        Scomplex_ /= nsamp;

        // Adding diagonal shift
        Scomplex_ += Eigen::MatrixXd::Identity(npar, npar) * sr_diag_shift_;

        if (use_cholesky_ == false) {
          Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> qr(npar, npar);
          qr.setThreshold(1.0e-6);
          qr.compute(Scomplex_);
          deltaP = qr.solve(grad);
        } else {
          Eigen::LLT<Eigen::MatrixXcd> llt(npar);
          llt.compute(Scomplex_);
          deltaP = llt.solve(grad);
        }
      } else {
        Eigen::ConjugateGradient<SrMatrixComplex, Eigen::Lower | Eigen::Upper,
                                 Eigen::IdentityPreconditioner>
            it_solver;
        // Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>
        // it_solver;
        it_solver.setTolerance(1.0e-3);
        SrMatrixComplex S;
        S.attachMatrix(Oks);
        S.setShift(sr_diag_shift_);
        S.setScale(1. / nsamp);

        it_solver.compute(S);
        deltaP = it_solver.solve(grad);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    } else {
      if (!use_iterative_) {
        // Explicit construction of the S matrix
        Sreal_.resize(npar, npar);
        Sreal_ = (Oks.adjoint() * Oks).real();
        SumOnNodes(Sreal_);
        Sreal_ /= nsamp;

        // Adding diagonal shift
        Sreal_ += Eigen::MatrixXd::Identity(npar, npar) * sr_diag_shift_;

        if (use_cholesky_ == false) {
          Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(npar, npar);
          qr.setThreshold(1.0e-6);
          qr.compute(Sreal_);
          deltaP.real() = qr.solve(grad.real());
        } else {
          Eigen::LLT<Eigen::MatrixXd> llt(npar);
          llt.compute(Sreal_);
          deltaP.real() = llt.solve(grad.real());
          deltaP.imag().setZero();
        }
      } else {
        Eigen::ConjugateGradient<SrMatrixReal, Eigen::Lower | Eigen::Upper,
                                 Eigen::IdentityPreconditioner>
            it_solver;
        // Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>
        // it_solver;
        it_solver.setTolerance(1.0e-3);
        SrMatrixReal S;
        S.attachMatrix(Oks);
        S.setShift(sr_diag_shift_);
        S.setScale(1. / nsamp);

        it_solver.compute(S);
        deltaP.real() = it_solver.solve(grad.real());
        deltaP.imag().setZero();
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }

  void setDefaultParameters() {
    sr_diag_shift_ = 0.01;
    use_iterative_ = false;
    use_cholesky_ = true;
    is_holomorphic_ = true;
  }

  void setParameters(double diagshift = 0.01, bool use_iterative = false,
                     bool use_cholesky = true, bool is_holomorphic = true) {
    sr_diag_shift_ = diagshift;
    use_iterative_ = use_iterative;
    use_cholesky_ = use_cholesky;
    is_holomorphic_ = is_holomorphic;

    InfoMessage() << "Using the Stochastic reconfiguration method" << std::endl;

    if (use_iterative_) {
      InfoMessage() << "With iterative solver" << std::endl;
    } else {
      if (use_cholesky_) {
        InfoMessage() << "Using Cholesky decomposition" << std::endl;
      }
    }
  }
};  // namespace netket

}  // namespace netket

#endif
