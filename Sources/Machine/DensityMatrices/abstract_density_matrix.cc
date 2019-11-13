//
// Created by Filippo Vicentini on 07/11/2019.
//

#include "abstract_density_matrix.hpp"

namespace netket {
using VectorType = AbstractDensityMatrix::VectorType;
// using Edge = AbstractGraph::Edge;

std::shared_ptr<const AbstractHilbert>
AbstractDensityMatrix::GetHilbertPhysicalShared() const {
  return std::static_pointer_cast<const DoubledHilbert>(GetHilbertShared())->GetHilbertPhysicalShared();
}

const AbstractHilbert &AbstractDensityMatrix::GetHilbertPhysical() const
    noexcept {
  return *GetHilbertPhysicalShared();
}

void AbstractDensityMatrix::LogVal(Eigen::Ref<const RowMatrix<double>> v,
                                   Eigen::Ref<VectorType> out, const any &lt) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, 2 * NvisiblePhysical()});
  CheckShape(__FUNCTION__, "out", out.size(), out.rows());
  LogVal(v.block(0, 0, v.rows(), NvisiblePhysical()),
         v.block(0, NvisiblePhysical(), v.rows(), NvisiblePhysical()), out, lt);
}

Complex AbstractDensityMatrix::LogValSingle(VisibleConstType v, const any &lt) {
  CheckShape(__FUNCTION__, "v", {v.rows()}, {2 * NvisiblePhysical()});

  return LogValSingle(v.head(NvisiblePhysical()), v.tail(NvisiblePhysical()), lt);
}

// Batched version of LogVal
void AbstractDensityMatrix::LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                                   Eigen::Ref<const RowMatrix<double>> vc,
                                   Eigen::Ref<VectorType> out, const any &lt) {
  CheckShape(__FUNCTION__, "vr", {vr.rows(), vr.cols()},
             {std::ignore, NvisiblePhysical()});
  CheckShape(__FUNCTION__, "vc", {vc.rows(), vc.cols()},
             {std::ignore, NvisiblePhysical()});
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
  CheckShape(__FUNCTION__, "v", {v.rows()}, {2 * NvisiblePhysical()});
  return this->DerLogSingle(v.head(NvisiblePhysical()), v.tail(NvisiblePhysical()), cache);
};

// I have no idea why this gives a linker error if it is not defined.
// Anyhow, It should never be called.
VectorType AbstractDensityMatrix::DerLogSingle(VisibleConstType vr,
                                               VisibleConstType vc,
                                               const any &cache) {
  std::cout << "Executed code that should not be executed." << std::endl;
  throw;
};

// I have no idea why this gives a linker error if it is not defined.
// Anyhow, It should never be called.
Complex AbstractDensityMatrix::LogValSingle(VisibleConstType vr,
                                            VisibleConstType vc,
                                            const any &cache) {
  std::cout << "Executed code that should not be executed." << std::endl;
  throw;
};

// Batched version of DerLog
void AbstractDensityMatrix::DerLog(Eigen::Ref<const RowMatrix<double>> v,
                                   Eigen::Ref<RowMatrix<Complex>> out,
                                   const any &cache) {
  CheckShape(__FUNCTION__, "vr", {v.rows(), v.cols()},
             {std::ignore, 2 * NvisiblePhysical()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()}, {v.rows(), Npar()});

  DerLog(v.block(0, 0, v.rows(), NvisiblePhysical()),
         v.block(0, NvisiblePhysical(), v.rows(), NvisiblePhysical()), out, cache);
}

void AbstractDensityMatrix::DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                                   Eigen::Ref<const RowMatrix<double>> vc,
                                   Eigen::Ref<RowMatrix<Complex>> out,
                                   const any &cache) {
  CheckShape(__FUNCTION__, "vr", {vr.rows(), vr.cols()},
             {std::ignore, NvisiblePhysical()});
  CheckShape(__FUNCTION__, "vc", {vc.rows(), vc.cols()},
             {std::ignore, NvisiblePhysical()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()},
             {vr.rows(), Npar()});

  for (auto i = Index{0}; i < vr.rows(); ++i) {
    out.row(i) =
        DerLogSingle(vr.row(i), vc.row(i), cache);
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