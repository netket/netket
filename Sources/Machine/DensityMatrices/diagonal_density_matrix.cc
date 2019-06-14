//
// Created by Filippo Vicentini on 2019-06-14.
//

#include "diagonal_density_matrix.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

namespace netket {

using VectorType = DiagonalDensityMatrix::VectorType;
using VisibleConstType = DiagonalDensityMatrix::VisibleConstType;

DiagonalDensityMatrix::DiagonalDensityMatrix(AbstractDensityMatrix &dm)
    : AbstractMachine(dm.GetHilbertPhysicalShared()), density_matrix_(dm) {}


    
Complex DiagonalDensityMatrix::LogVal(VisibleConstType v) {
  return density_matrix_.LogVal(v, v);
}

Complex DiagonalDensityMatrix::LogVal(VisibleConstType v,
                                      const LookupType &lt) {
  return density_matrix_.LogVal(v, v, lt);
}

void DiagonalDensityMatrix::InitLookup(VisibleConstType v, LookupType &lt) {
  return density_matrix_.InitLookup(v, v, lt);
}

void DiagonalDensityMatrix::UpdateLookup(VisibleConstType v,
                                         const std::vector<int> &tochange,
                                         const std::vector<double> &newconf,
                                         LookupType &lt) {
  return density_matrix_.UpdateLookup(v, v, tochange, tochange, newconf,
                                      newconf, lt);
}

VectorType DiagonalDensityMatrix::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  return density_matrix_.LogValDiff(v, v, tochange, tochange, newconf, newconf);
}

Complex DiagonalDensityMatrix::LogValDiff(VisibleConstType v,
                                          const std::vector<int> &tochange,
                                          const std::vector<double> &newconf,
                                          const LookupType &lt) {
  return density_matrix_.LogValDiff(v, v, tochange, tochange, newconf, newconf,
                                    lt);
}

VectorType DiagonalDensityMatrix::DerLog(VisibleConstType v) {
  return density_matrix_.DerLog(v, v);
}

VectorType DiagonalDensityMatrix::DerLog(VisibleConstType v,
                                         const LookupType &lt) {
  return density_matrix_.DerLog(v, v, lt);
}

VectorType DiagonalDensityMatrix::DerLogChanged(
    VisibleConstType v, const std::vector<int> &tochange,
    const std::vector<double> &newconf) {
  return density_matrix_.DerLogChanged(v, v, tochange, tochange, newconf, newconf);
}

}  // namespace netket