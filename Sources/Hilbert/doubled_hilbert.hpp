//
// Created by Filippo Vicentini on 12/11/2019.
//

#ifndef NETKET_DOUBLED_HILBERT_HPP
#define NETKET_DOUBLED_HILBERT_HPP

#include <utility>

#include "Graph/abstract_graph.hpp"
#include "Graph/doubled_graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"

namespace netket {

class DoubledHilbert : public AbstractHilbert {
  std::unique_ptr<const AbstractGraph> graph_doubled_;
  std::shared_ptr<const AbstractHilbert> hilbert_physical_;

  int size_;

 protected:
  DoubledHilbert(std::shared_ptr<const AbstractHilbert> hilbert,
                 std::unique_ptr<const AbstractGraph> doubled_graph);

 public:
  explicit DoubledHilbert(const std::shared_ptr<const AbstractHilbert>& hilbert)
      : DoubledHilbert(hilbert, DoubledGraph(hilbert->GetGraph())){};

  // -- DoubledHilbert Interface  -- //
  int SizePhysical() const;
  const AbstractGraph &GetGraphPhysical() const noexcept;
  const AbstractHilbert &GetHilbertPhysical() const noexcept;
  std::shared_ptr<const AbstractHilbert> GetHilbertPhysicalShared() const;

  void RandomValsRows(Eigen::Ref<Eigen::VectorXd> state,
                      netket::default_random_engine &rgen) const;
  void RandomValsCols(Eigen::Ref<Eigen::VectorXd> state,
                      netket::default_random_engine &rgen) const;
  void RandomValsPhysical(Eigen::Ref<Eigen::VectorXd> state,
                          default_random_engine &rgen) const;

  void UpdateConfPhysical(Eigen::Ref<Eigen::VectorXd> v,
                          nonstd::span<const int> tochange,
                          nonstd::span<const double> newconf) const;

  // -- AbstractHilbert interface -- //

  bool IsDiscrete() const override;
  int LocalSize() const override;
  int Size() const override;
  std::vector<double> LocalStates() const override;
  const AbstractGraph &GetGraph() const noexcept override;

  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const override;

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  nonstd::span<const int> tochange,
                  nonstd::span<const double> newconf) const override;
};
}  // namespace netket

#endif  // NETKET_DOUBLED_HILBERT_HPP
