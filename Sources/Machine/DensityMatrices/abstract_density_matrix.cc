//
// Created by Filippo Vicentini on 2019-06-14.
//

#include "abstract_density_matrix.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

namespace netket {
using VectorType = AbstractDensityMatrix::VectorType;
using VisibleConstType = AbstractDensityMatrix::VisibleConstType;
using LookupType = AbstractDensityMatrix::LookupType;
using Edge = AbstractGraph::Edge;
using ChangeInfo = AbstractDensityMatrix::ChangeInfo;
using RowColChangeInfo = AbstractDensityMatrix::RowColChangeInfo;

AbstractDensityMatrix::AbstractDensityMatrix(
    std::shared_ptr<const AbstractHilbert> physical_hilbert,
    std::unique_ptr<const AbstractGraph> doubled_graph)
    : AbstractMachine(std::make_shared<CustomHilbert>(
          *doubled_graph, physical_hilbert->LocalStates())),
      hilbert_physical_(physical_hilbert),
      graph_doubled_(std::move(doubled_graph)) {}

AbstractDensityMatrix::AbstractDensityMatrix(
    std::shared_ptr<const AbstractHilbert> hilbert)
    : AbstractDensityMatrix(hilbert, DoubledGraph(hilbert->GetGraph())) {}

std::shared_ptr<const AbstractHilbert>
AbstractDensityMatrix::GetHilbertPhysicalShared() const {
  return hilbert_physical_;
}

const AbstractHilbert &AbstractDensityMatrix::GetHilbertPhysical() const
    noexcept {
  return *hilbert_physical_;
}

void AbstractDensityMatrix::InitLookup(VisibleConstType v, LookupType &lt) {
  // split into first half and second half
  VisibleConstType v_r = v.head(GetHilbertPhysical().Size());
  VisibleConstType v_c = v.tail(GetHilbertPhysical().Size());

  return InitLookup(v_r, v_c, lt);
}

void AbstractDensityMatrix::UpdateLookup(VisibleConstType v,
                                         const std::vector<int> &tochange,
                                         const std::vector<double> &newconf,
                                         LookupType &lt) {
  auto split_data = SplitRowColsChange(tochange, newconf);
  auto tochange_r = std::move(std::get<0>(std::get<0>(split_data)));
  auto tochange_c = std::move(std::get<0>(std::get<1>(split_data)));
  auto newconf_r = std::move(std::get<1>(std::get<0>(split_data)));
  auto newconf_c = std::move(std::get<1>(std::get<1>(split_data)));

  // split into first half and second half
  VisibleConstType v_r = v.head(GetHilbertPhysical().Size());
  VisibleConstType v_c = v.tail(GetHilbertPhysical().Size());
  return UpdateLookup(v_r, v_c, tochange_r, tochange_c, newconf_r, newconf_c,
                      lt);
}

VectorType AbstractDensityMatrix::DerLog(VisibleConstType v) {
  return DerLog(v.head(GetHilbertPhysical().Size()),
                v.tail(GetHilbertPhysical().Size()));
}

VectorType AbstractDensityMatrix::DerLog(VisibleConstType v,
                                         const LookupType &lt) {
  return DerLog(v.head(GetHilbertPhysical().Size()),
                v.tail(GetHilbertPhysical().Size()), lt);
}

Complex AbstractDensityMatrix::LogVal(VisibleConstType v) {
  return LogVal(v.head(GetHilbertPhysical().Size()),
                v.tail(GetHilbertPhysical().Size()));
}

Complex AbstractDensityMatrix::LogVal(VisibleConstType v,
                                      const LookupType &lt) {
  return LogVal(v.head(GetHilbertPhysical().Size()),
                v.tail(GetHilbertPhysical().Size()), lt);
}

VectorType AbstractDensityMatrix::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  VisibleConstType v_r = v.head(GetHilbertPhysical().Size());
  VisibleConstType v_c = v.tail(GetHilbertPhysical().Size());

  std::vector<std::vector<int>> tochange_all_r, tochange_all_c;
  std::vector<std::vector<double>> newconf_all_r, newconf_all_c;

  const std::size_t nconn = tochange.size();
  for (std::size_t k = 0; k < nconn; k++) {
    auto split_data = SplitRowColsChange(tochange[k], newconf[k]);
    tochange_all_r.emplace_back(
        std::move(std::get<0>(std::get<0>(split_data))));
    tochange_all_c.emplace_back(
        std::move(std::get<0>(std::get<1>(split_data))));
    newconf_all_r.emplace_back(std::move(std::get<1>(std::get<0>(split_data))));
    newconf_all_c.emplace_back(std::move(std::get<1>(std::get<1>(split_data))));
  }

  return LogValDiff(v_r, v_c, tochange_all_r, tochange_all_c, newconf_all_r,
                    newconf_all_c);
}

Complex AbstractDensityMatrix::LogValDiff(VisibleConstType v,
                                          const std::vector<int> &tochange,
                                          const std::vector<double> &newconf,
                                          const LookupType &lt) {
  VisibleConstType v_r = v.head(GetHilbertPhysical().Size());
  VisibleConstType v_c = v.tail(GetHilbertPhysical().Size());

  auto split_data = SplitRowColsChange(tochange, newconf);
  auto tochange_r = std::move(std::get<0>(std::get<0>(split_data)));
  auto tochange_c = std::move(std::get<0>(std::get<1>(split_data)));
  auto newconf_r = std::move(std::get<1>(std::get<0>(split_data)));
  auto newconf_c = std::move(std::get<1>(std::get<1>(split_data)));

  return LogValDiff(v_r, v_c, tochange_r, tochange_c, newconf_r, newconf_c, lt);
}

VectorType AbstractDensityMatrix::DerLogChanged(
    VisibleConstType v_r, VisibleConstType v_c,
    const std::vector<int> &tochange_r, const std::vector<int> &tochange_c,
    const std::vector<double> &newconf_r,
    const std::vector<double> &newconf_c) {
  VisibleType vp_r(v_r);
  VisibleType vp_c(v_c);
  hilbert_physical_->UpdateConf(vp_r, tochange_r, newconf_r);
  hilbert_physical_->UpdateConf(vp_c, tochange_c, newconf_c);
  return DerLog(vp_r, vp_c);
}

RowColChangeInfo AbstractDensityMatrix::SplitRowColsChange(
    const std::vector<int> &tochange,
    const std::vector<double> &newconf) const {
  std::vector<int> tochange_r, tochange_c;
  std::vector<double> newconf_r, newconf_c;

  if (tochange.size() != 0) {
    for (std::size_t s = 0; s < tochange.size(); s++) {
      const int sf = tochange[s];
      if (sf < Nvisible()) {
        tochange_r.push_back(sf);
        newconf_r.push_back(newconf[s]);
      } else {
        tochange_c.push_back(sf - Nvisible());
        newconf_c.push_back(newconf[s]);
      }
    }
  }

  ChangeInfo info_r(std::move(tochange_r), std::move(newconf_r));
  ChangeInfo info_c(std::move(tochange_c), std::move(newconf_c));
  return RowColChangeInfo(std::move(info_r), std::move(info_c));
}

std::unique_ptr<CustomGraph> AbstractDensityMatrix::DoubledGraph(
    const AbstractGraph &graph) {
  auto n_sites = graph.Nsites();
  std::vector<Edge> d_edges(graph.Edges().size());
  auto eclist = graph.EdgeColors();

  // same graph
  auto d_eclist = graph.EdgeColors();
  for (auto edge : graph.Edges()) {
    d_edges.push_back(edge);
  }

  // second graph
  for (auto edge : graph.Edges()) {
    Edge new_edge = edge;
    new_edge[0] += n_sites;
    new_edge[1] += n_sites;

    d_edges.push_back(new_edge);
    d_eclist.emplace(new_edge, eclist[edge]);
  }

  std::vector<std::vector<int>> d_automorphisms;
  for (auto automorphism : graph.SymmetryTable()) {
    std::vector<int> d_automorphism = automorphism;
    for (auto s : automorphism) {
      d_automorphism.push_back(s + n_sites);
    }
    d_automorphisms.push_back(d_automorphism);
  }

  return make_unique<CustomGraph>(
      CustomGraph(d_edges, d_eclist, d_automorphisms));
}

}  // namespace netket