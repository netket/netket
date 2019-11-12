//
// Created by Filippo Vicentini on 07/11/2019.
//

#include "abstract_density_matrix.hpp"

namespace netket {
using VectorType = AbstractDensityMatrix::VectorType;
// using Edge = AbstractGraph::Edge;

std::shared_ptr<const AbstractHilbert>
AbstractDensityMatrix::GetHilbertPhysicalShared() const {
  return hilbert_physical_;
}

const AbstractHilbert &AbstractDensityMatrix::GetHilbertPhysical() const
    noexcept {
  return *hilbert_physical_;
}

void AbstractDensityMatrix::LogVal(Eigen::Ref<const RowMatrix<double>> v,
                                   Eigen::Ref<VectorType> out, const any &lt) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, 2 * Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), out.rows());
  LogVal(v.block(0, 0, v.rows(), Nvisible()),
         v.block(0, Nvisible(), v.rows(), Nvisible()), out, lt);
}

Complex AbstractDensityMatrix::LogValSingle(VisibleConstType v, const any &lt) {
  CheckShape(__FUNCTION__, "v", {v.rows()}, {2 * Nvisible()});

  return LogValSingle(v.head(Nvisible()), v.tail(Nvisible()), lt);
}

// Batched version of LogVal
void AbstractDensityMatrix::LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                                   Eigen::Ref<const RowMatrix<double>> vc,
                                   Eigen::Ref<VectorType> out, const any &lt) {
  CheckShape(__FUNCTION__, "vr", {vr.rows(), vr.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "vc", {vc.rows(), vc.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), out.rows());

  for (auto i = Index{0}; i < vr.rows(); ++i) {
    out(i) = LogValSingle(vr.row(i), vc.row(i), lt);
  }
}

VectorType AbstractDensityMatrix::LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                                         Eigen::Ref<const RowMatrix<double>> vc,
                                         const any &cache) {
  VectorType out(vr.rows());
  LogVal(vr, vc, out, cache);
  return out;
}

VectorType AbstractDensityMatrix::DerLogSingle(VisibleConstType v,
                                               const any &cache) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, 2 * Nvisible()});
  return this->DerLogSingle(v.head(Nvisible()), v.tail(Nvisible()), cache);
};

// I have no idea why this gives a linker error if it is not defined.
// Anyhow, It should never be called.
VectorType AbstractDensityMatrix::DerLogSingle(VisibleConstType vr,
                                               VisibleConstType vc,
                                               const any &cache) {
  throw;
};

// I have no idea why this gives a linker error if it is not defined.
// Anyhow, It should never be called.
Complex AbstractDensityMatrix::LogValSingle(VisibleConstType vr,
                                            VisibleConstType vc,
                                            const any &cache) {
  throw;
};

// Batched version of DerLog
void AbstractDensityMatrix::DerLog(Eigen::Ref<const RowMatrix<double>> v,
                                   Eigen::Ref<RowMatrix<Complex>> out,
                                   const any &cache) {
  CheckShape(__FUNCTION__, "vr", {v.rows(), v.cols()},
             {std::ignore, 2 * Nvisible()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()}, {v.rows(), Npar()});
  DerLog(v.block(0, 0, v.rows(), Nvisible()),
         v.block(0, Nvisible(), v.rows(), Nvisible()), out, cache);
}

void AbstractDensityMatrix::DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                                   Eigen::Ref<const RowMatrix<double>> vc,
                                   Eigen::Ref<RowMatrix<Complex>> out,
                                   const any &cache) {
  CheckShape(__FUNCTION__, "vr", {vr.rows(), vr.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "vc", {vc.rows(), vc.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()},
             {vr.rows(), Npar()});
  for (auto i = Index{0}; i < vr.rows(); ++i) {
    out.row(i) =
        AbstractDensityMatrix::DerLogSingle(vr.row(i), vc.row(i), cache);
  }
}

RowMatrix<Complex> AbstractDensityMatrix::DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                                   Eigen::Ref<const RowMatrix<double>> vc,
                                   const linb::any &cache) {
  RowMatrix<Complex> out(vr.rows(), Npar());
  DerLog(vr, vc, out, cache);
  return out;
}

}  // namespace netket