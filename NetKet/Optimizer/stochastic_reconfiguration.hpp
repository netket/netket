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

  Eigen::MatrixXd S_;

 public:
  SR() { setDefaultParameters(); }

  void ComputeUpdate(const Eigen::Ref<const Eigen::MatrixXcd> Oks,
                     const Eigen::Ref<const Eigen::VectorXd> grad,
                     Eigen::Ref<Eigen::VectorXd> deltaP) {
    double nsamp = Oks.rows();

    SumOnNodes(nsamp);
    auto npar = grad.size();

    if (!use_iterative_) {
      // Explicit construction of the S matrix
      S_.resize(npar, npar);
      S_ = (Oks.adjoint() * Oks).real();
      SumOnNodes(S_);
      S_ /= nsamp;

      // Adding diagonal shift
      S_ += Eigen::MatrixXd::Identity(npar, npar) * sr_diag_shift_;

      if (use_cholesky_ == false) {
        Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(npar, npar);
        qr.setThreshold(1.0e-6);
        qr.compute(S_);
        deltaP = qr.solve(grad);
      } else {
        Eigen::LLT<Eigen::MatrixXd> llt(npar);
        llt.compute(S_);
        deltaP = llt.solve(grad);
      }
      // Eigen::VectorXcd deltaP=S.jacobiSvd(ComputeThinU |
      // ComputeThinV).solve(b);

    } else {
      Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper,
                               Eigen::IdentityPreconditioner>
          it_solver;
      // Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>
      // it_solver;
      it_solver.setTolerance(1.0e-3);
      MatrixReplacement S;
      S.attachMatrix(Oks);
      S.setShift(sr_diag_shift_);
      S.setScale(1. / nsamp);

      it_solver.compute(S);
      deltaP = it_solver.solve(grad);

      // if(mynode_==0){
      //   std::cerr<<it_solver.iterations()<<"
      //   "<<it_solver.error()<<std::endl;
      // }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  void setDefaultParameters() {
    sr_diag_shift_ = 0.01;
    use_iterative_ = false;
    use_cholesky_ = true;
  }

  void setParameters(double diagshift = 0.01, bool use_iterative = false,
                     bool use_cholesky = true) {
    sr_diag_shift_ = diagshift;
    use_iterative_ = use_iterative;
    use_cholesky_ = use_cholesky;

    InfoMessage() << "Using the Stochastic reconfiguration method" << std::endl;

    if (use_iterative_) {
      InfoMessage() << "With iterative solver" << std::endl;
    } else {
      if (use_cholesky_) {
        InfoMessage() << "Using Cholesky decomposition" << std::endl;
      }
    }
  }
};

}  // namespace netket

#endif
