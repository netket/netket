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

#ifndef NETKET_NDM_SPIN_PHASE_HPP
#define NETKET_NDM_SPIN_PHASE_HPP

#include "abstract_density_matrix.hpp"

namespace netket {

/** Neural Density Matrix machine class with spin 1/2 hidden units.
This version has real-valued weights and two NDMs parameterizing phase and
amplitude
 *
 */
class NdmSpinPhase : public AbstractDensityMatrix {
  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of ancillary units
  int na_;

  // number of parameters
  int npar_{};

  // visible units bias
  RealVectorType b1_;
  RealVectorType b2_;

  // hidden units bias
  RealVectorType h1_;
  RealVectorType h2_;

  // ancillary units bias
  RealVectorType d1_;

  // hidden unit weights
  RealMatrixType W1_;
  RealMatrixType W2_;

  // ancillary unit weights
  RealMatrixType U1_;
  RealMatrixType U2_;

  // Caches
  RealMatrixType thetas_r1_;
  RealMatrixType thetas_r2_;
  RealMatrixType thetas_c1_;
  RealMatrixType thetas_c2_;
  RealMatrixType lnthetas_r1_;
  RealMatrixType lnthetas_r2_;
  RealMatrixType lnthetas_c1_;
  RealMatrixType lnthetas_c2_;
  RealMatrixType thetasnew_r1_;
  RealMatrixType thetasnew_r2_;
  RealMatrixType thetasnew_c1_;
  RealMatrixType thetasnew_c2_;
  RealMatrixType lnthetasnew_r1_;
  RealMatrixType lnthetasnew_r2_;
  RealMatrixType lnthetasnew_c1_;
  RealMatrixType lnthetasnew_c2_;

  MatrixType     thetas_a_;
  MatrixType     lnthetas_a_;
  RealMatrixType thetas_a1_;
  RealMatrixType thetas_a2_;
  RealMatrixType thetasnew_a1_;
  RealMatrixType thetasnew_a2_;
  MatrixType pi_;
  MatrixType lnpi_;
  MatrixType lnpinew_;

  MatrixType vsum_;
  MatrixType vdelta_;



  bool useb_;
  bool useh_;
  bool used_;

  const Complex I_;

  using LookupType = std::vector<RealVectorType>;

 public:
  explicit NdmSpinPhase(std::shared_ptr<const AbstractHilbert> hilbert,
                        int nhidden = 0, int nancilla = 0, int alpha = 0,
                        int beta = 0, bool useb = true, bool useh = true,
                        bool used = true)
      : AbstractDensityMatrix(hilbert),
        nv_(hilbert->Size()),
        useb_(useb),
        useh_(useh),
        used_(used),
        I_(0, 1) {
    nh_ = std::max(nhidden, alpha * nv_);
    na_ = std::max(nancilla, beta * nv_);
    Init();
  }

  void Init();

  /*int Nvisible() const override { return nv_; }*/
  int NvisiblePhysical() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Nancilla() const { return na_; }

  int Npar() const override { return npar_; }

  VectorType GetParameters() override;

  void SetParameters(VectorConstRefType pars) override;

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogValSingle(VisibleConstType vr, VisibleConstType vc,
                       const any & lookup) override;

  void LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                            Eigen::Ref<const RowMatrix<double>> vc,
                    Eigen::Ref<VectorType> out,
                            const any & lup) override;

  VectorType DerLogSingle(VisibleConstType vr, VisibleConstType vc,
                          const any &cache) override;

  virtual void DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                      Eigen::Ref<const RowMatrix<double>> vc,
                      Eigen::Ref<RowMatrix<Complex>> out,
                      const any &cache) override;

  void Save(const std::string &filename) const override;

  void Load(const std::string &filename) override;

  bool IsHolomorphic() const noexcept override;

  Index BatchSize() const noexcept;
  void BatchSize(Index batch_size);

};  // namespace netket
}  // namespace netket
#endif  // NETKET_NDM_SPIN_PHASE_HPP
