//
// Created by Filippo Vicentini on 12/11/2019.
//

#include "doubled_hilbert.hpp"

namespace netket {

DoubledHilbert::DoubledHilbert(
    std::shared_ptr<const AbstractHilbert> hilbert,
    std::unique_ptr<const AbstractGraph> doubled_graph)
    : hilbert_physical_(std::move(hilbert)),
      graph_doubled_(std::move(doubled_graph)) {
  size_ = graph_doubled_->Size();
}

bool DoubledHilbert::IsDiscrete() const {
  return hilbert_physical_->IsDiscrete();
}

int DoubledHilbert::LocalSize() const { return hilbert_physical_->LocalSize(); }

int DoubledHilbert::Size() const {
  return size_;  // it is 2 times the number of physical sites
}

int DoubledHilbert::SizePhysical() const { return hilbert_physical_->Size(); }

std::vector<double> DoubledHilbert::LocalStates() const {
  return hilbert_physical_->LocalStates();
}

void DoubledHilbert::RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                                netket::default_random_engine& rgen) const {
  assert(state.size() == size_);

  auto N = SizePhysical();
  hilbert_physical_->RandomVals(state.head(N), rgen);
  hilbert_physical_->RandomVals(state.tail(N), rgen);
}

void DoubledHilbert::UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                                nonstd::span<const int> tochange,
                                nonstd::span<const double> newconf) const {
  assert(v.size() == size_);

  int i = 0;
  for (auto sf : tochange) {
    v(sf) = newconf[i];
    i++;
  }
}

void DoubledHilbert::RandomValsRows(Eigen::Ref<Eigen::VectorXd> state,
                                    netket::default_random_engine& rgen) const {
  assert(state.size() == size_);

  hilbert_physical_->RandomVals(state.head(SizePhysical()), rgen);
}

void DoubledHilbert::RandomValsCols(Eigen::Ref<Eigen::VectorXd> state,
                                    netket::default_random_engine& rgen) const {
  assert(state.size() == size_);

  hilbert_physical_->RandomVals(state.tail(SizePhysical()), rgen);
}

void DoubledHilbert::RandomValsPhysical(
    Eigen::Ref<Eigen::VectorXd> state,
    netket::default_random_engine& rgen) const {
  assert(state.size() == size_);

  hilbert_physical_->RandomVals(state, rgen);
}

void DoubledHilbert::UpdateConfPhysical(
    Eigen::Ref<Eigen::VectorXd> v, nonstd::span<const int> tochange,
    nonstd::span<const double> newconf) const {
  hilbert_physical_->UpdateConf(v, tochange, newconf);
}

const AbstractHilbert& DoubledHilbert::GetHilbertPhysical() const noexcept {
  return *hilbert_physical_;
}

std::shared_ptr<const AbstractHilbert>
DoubledHilbert::GetHilbertPhysicalShared() const {
  return hilbert_physical_;
}

const AbstractGraph& DoubledHilbert::GetGraphPhysical() const noexcept {
  return hilbert_physical_->GetGraph();
}

const AbstractGraph& DoubledHilbert::GetGraph() const noexcept {
  return *graph_doubled_;
}

}  // namespace netket