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

#include "Sampler/metropolis_local_v2.hpp"

namespace netket {

namespace detail {
void Flipper::RandomState() {
  std::generate(state_.data(), state_.data() + state_.size(), [this]() {
    return local_states_[std::uniform_int_distribution<int>{
        0, static_cast<int>(local_states_.size()) - 1}(Generator())];
  });
}

void Flipper::RandomSites() {
  std::generate(sites_.data(), sites_.data() + sites_.size(), [this]() {
    return std::uniform_int_distribution<Index>{0,
                                                SystemSize() - 1}(Generator());
  });
}

void Flipper::RandomValues() {
  // `g` proposes new value for spin `sites_(j)` in Markov chain `j`. There
  // are `local_states_.size() - 1` possible values (minus one is because we
  // don't want to stay in the same state). Thus first, we generate a random
  // number in `[0, local_states_.size() - 2]`. Next step is to transform the
  // result to avoid the gap. Here's an example:
  //
  // ```
  //    indices         0 1 2 3
  //                   +-+-+-+-+
  //    local_states_  | | |X| |
  //                   +-+-+-+-+
  //    transformed
  //      indices       0 1   2
  //
  // ```
  // `X` denotes the current state. We see that transformed index is equal to
  // the original one for all positions before `X`. After `X` however, we need
  // to increment indices by 1.
  const auto g = [this](const int j) {
    const auto idx = std::uniform_int_distribution<int>{
        0, static_cast<int>(local_states_.size()) - 2}(engine_.Get());
    return local_states_[idx + (local_states_[idx] >= state_(j, sites_(j)))];
  };
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    values_(j) = g(j);
  }
}

Flipper::Flipper(std::pair<Index, Index> const shape,
                 std::vector<double> local_states)
    : sites_{},
      values_{},
      state_{},
      local_states_{std::move(local_states)},
      proposed_{},
      engine_{} {
  Index batch_size, system_size;
  std::tie(batch_size, system_size) = shape;
  if (batch_size < 1) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size << "; expected >=1";
    throw InvalidInputError{msg.str()};
  }
  if (system_size < 1) {
    std::ostringstream msg;
    msg << "invalid system size: " << system_size << "; expected >=1";
    throw InvalidInputError{msg.str()};
  }
  if (local_states_.empty()) {
    throw InvalidInputError{"invalid local states: []"};
  }

  // #RandomValues() relies on the fact that locat_states_ are sorted.
  std::sort(local_states_.begin(), local_states_.end());

  sites_.resize(batch_size);
  values_.resize(batch_size);
  state_.resize(batch_size, system_size);
  proposed_.resize(batch_size);
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    proposed_[j].values = {&values_(j), 1};
    proposed_[j].sites = {&sites_(j), 1};
  }
  Reset();
}

void Flipper::Reset() {
  RandomState();
  RandomSites();
  RandomValues();
}

void Flipper::Next(nonstd::span<const bool> accept) {
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    if (accept[j]) {
      state_(j, sites_(j)) = values_(j);
    }
  }
  RandomSites();
  RandomValues();
}

nonstd::span<Suggestion const> Flipper::Read() const noexcept {
  using span = nonstd::span<Suggestion const>;
  return span{proposed_.data(),
              static_cast<span::index_type>(proposed_.size())};
}

const RowMatrix<double>& Flipper::Current() const noexcept { return state_; }

void Flipper::Read(Eigen::Ref<RowMatrix<double>> x) const noexcept {
  assert(x.rows() == BatchSize() && x.cols() == SystemSize());
  x = state_;
  const auto updates = Read();
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    const auto& suggestion = updates[j];
    assert(suggestion.sites.size() == 1 && suggestion.values.size() == 1);
    x(j, suggestion.sites[0]) = suggestion.values[0];
  }
}

Index CheckBatchSize(const Index batch_size) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  return batch_size;
}

}  // namespace detail

MetropolisLocalV2::MetropolisLocalV2(RbmSpinV2& machine, const Index batch_size,
                                     std::true_type /*safe*/)
    : machine_{machine},
      flipper_{{batch_size, machine.Nvisible()},
               machine.GetHilbert().LocalStates()},
      proposed_X_(batch_size, machine.Nvisible()),
      proposed_Y_(batch_size),
      current_Y_(batch_size),
      randoms_(batch_size),
      accept_(batch_size) {
  machine_.LogVal(flipper_.Current(), current_Y_, {});
}

MetropolisLocalV2::MetropolisLocalV2(RbmSpinV2& machine, const Index batch_size)
    : MetropolisLocalV2{machine, detail::CheckBatchSize(batch_size), {}} {}

void MetropolisLocalV2::Reset() {
  flipper_.Reset();
  machine_.LogVal(flipper_.Current(), current_Y_, {});
}

std::pair<Eigen::Ref<const MetropolisLocalV2::InputType>,
          Eigen::Ref<const Eigen::VectorXcd>>
MetropolisLocalV2::Read() {
  return {flipper_.Current(), current_Y_};
}

void MetropolisLocalV2::Next() {
  flipper_.Read(proposed_X_);
  machine_.LogVal(proposed_X_, proposed_Y_, {});
  std::generate(randoms_.data(), randoms_.data() + randoms_.size(), [this]() {
    return std::uniform_real_distribution<double>{}(flipper_.Generator());
  });
  accept_ = randoms_ < (proposed_Y_ - current_Y_).exp().abs().square().min(1.0);
  current_Y_ = accept_.select(proposed_Y_, current_Y_);
  flipper_.Next({accept_});
}

void StepsRange::CheckValid() const {
  const auto error = [this](const char* expected) {
    std::ostringstream msg;
    msg << "invalid steps range: (start=" << start_ << ", end=" << end_
        << ", step=" << step_ << "); " << expected;
    throw InvalidInputError{msg.str()};
  };
  if (start_ < 0) error("expected start >= 0");
  if (end_ < 0) error("expected end >= 0");
  if (step_ <= 0) error("expected step >= 1");
  if (end_ < start_) error("expected start <= end");
}

namespace detail {
template <class Skip, class Record>
void LoopV2(StepsRange const& steps, Skip&& skip, Record&& record) {
  auto i = Index{0};
  // Skipping [0, 1, ..., start)
  for (; i < steps.start(); ++i) {
    skip();
  }
  // Record [start]
  record();
  for (i += steps.step(); i < steps.end(); i += steps.step()) {
    for (auto j = Index{1}; j < steps.step(); ++j) {
      skip();
    }
    record();
  }
}
}  // namespace detail

std::tuple<RowMatrix<double>, Eigen::VectorXcd,
           nonstd::optional<RowMatrix<Complex>>>
ComputeSamples(MetropolisLocalV2& sampler, StepsRange const& steps,
               bool compute_gradients) {
  sampler.Reset();
  const auto num_samples = steps.size() * sampler.BatchSize();

  RowMatrix<double> samples(num_samples, sampler.SystemSize());
  Eigen::VectorXcd values(num_samples);
  auto gradients =
      compute_gradients
          ? nonstd::optional<RowMatrix<Complex>>{nonstd::in_place, num_samples,
                                                 sampler.Machine().Npar()}
          : nonstd::nullopt;

  struct Skip {
    MetropolisLocalV2& sampler_;
    void operator()() const { sampler_.Next(); }
  };

  struct Record {
    MetropolisLocalV2& sampler_;
    RowMatrix<double>& samples_;
    VectorXcd& values_;
    nonstd::optional<RowMatrix<Complex>>& gradients_;
    Index i_;

    std::pair<Eigen::Ref<RowMatrix<double>>, Eigen::Ref<VectorXcd>> Batch() {
      const auto n = sampler_.BatchSize();
      return {samples_.block(i_ * n, 0, n, samples_.cols()),
              values_.segment(i_ * n, n)};
    }

    void Gradients() {
      const auto n = sampler_.BatchSize();
      const auto X = samples_.block(i_ * n, 0, n, samples_.cols());
      const auto out = gradients_->block(i_ * n, 0, n, gradients_->cols());
      sampler_.Machine().DerLog(X, out, any{});
    }

    void operator()() {
      assert(i_ * sampler_.BatchSize() < samples_.rows());
      Batch() = sampler_.Read();
      if (gradients_.has_value()) Gradients();
      if (++i_ * sampler_.BatchSize() != samples_.rows()) {
        sampler_.Next();
      }
    }
  };

  detail::LoopV2(steps, Skip{sampler},
                 Record{sampler, samples, values, gradients, 0});
  return std::make_tuple(std::move(samples), std::move(values),
                         std::move(gradients));
}

namespace detail {
struct Forward {
  Forward(RbmSpinV2& m, Index batch_size)
      : machine_{m},
        X_(batch_size, m.Nvisible()),
        Y_(batch_size),
        coeff_(batch_size),
        i_{0} {}

  bool Full() const noexcept { return i_ == BatchSize(); }
  Index BatchSize() const noexcept { return Y_.size(); }

  void Push(Eigen::Ref<const Eigen::VectorXd> v, const ConnectorRef& conn) {
    assert(!Full());
    X_.row(i_) = v.transpose();
    for (auto j = Index{0}; j < conn.tochange.size(); ++j) {
      X_(i_, conn.tochange[j]) = conn.newconf[j];
    }
    coeff_(i_) = conn.mel;
    ++i_;
  }

  void Fill(Eigen::Ref<const Eigen::VectorXd> v) {
    if (!Full()) {
      const auto n = BatchSize() - i_;
      X_.block(i_, 0, n, X_.cols()) = v.transpose().colwise().replicate(n);
      coeff_.segment(i_, n).setConstant(0.0);
      i_ += n;
    }
    assert(Full());
  }

  std::tuple<const Eigen::VectorXcd&, Eigen::VectorXcd&> Propagate() {
    assert(Full());
    machine_.LogVal(X_, /*out=*/Y_, /*cache=*/any{});
    i_ = 0;
    return std::tuple<const Eigen::VectorXcd&, Eigen::VectorXcd&>{coeff_, Y_};
  }

 private:
  RbmSpinV2& machine_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_;
  Eigen::VectorXcd Y_;
  Eigen::VectorXcd coeff_;
  Index i_;
};

struct Accumulator {
  Accumulator(Eigen::VectorXcd& loc, Forward& fwd)
      : locals_{loc}, index_{0}, accum_{0.0, 0.0}, forward_{fwd}, states_{} {
    states_.reserve(forward_.BatchSize());
  }

  void ProcessBatch() {
    assert(forward_.Full());
    auto result = forward_.Propagate();
    const auto& coeff = std::get<0>(result);
    auto& y = std::get<1>(result);

    // y <- log(ψ(s')) - log(ψ(s))
    {
      auto i = Index{0};
      for (auto j = size_t{0}; j < states_.size() - 1; ++j) {
        const auto& state = states_.at(j);
        for (auto n = Index{0}; n < state.first; ++n, ++i) {
          y(i) -= state.second;
        }
      }
      {
        const auto& state = states_.back();
        for (; i < y.size(); ++i) {
          y(i) -= state.second;
        }
      }
    }
    // y <- ψ(s')/ψ(s)
    y.array() = y.array().exp();

    assert(states_.size() > 0);
    auto i = Index{0};
    for (auto j = size_t{0}; j < states_.size() - 1; ++j) {
      const auto& state = states_[j];
      for (auto n = Index{0}; n < state.first; ++n, ++i) {
        accum_ += y(i) * coeff(i);
      }

      locals_(index_) = accum_;
      ++index_;
      accum_ = 0.0;
    }
    {
      const auto& state = states_.back();
      for (; i < y.size(); ++i) {
        accum_ += y(i) * coeff(i);
      }
    }
    if (states_.size() > 1) {
      std::swap(states_.front(), states_.back());
      states_.resize(1);
    }
    states_.front().first = 0;
  }

  void operator()(Eigen::Ref<const Eigen::VectorXd> v,
                  const ConnectorRef& conn) {
    assert(!forward_.Full());
    forward_.Push(v, conn);
    ++states_.back().first;
    if (forward_.Full()) {
      ProcessBatch();
    }
  }

  void operator()(Complex log_val) {
    assert(!forward_.Full());
    states_.emplace_back(0, log_val);
  }

  void Finalize(Eigen::Ref<const Eigen::VectorXd> v) {
    states_.emplace_back(0, states_.back().second);
    forward_.Fill(v);
    ProcessBatch();
  }

 private:
  Eigen::VectorXcd& locals_;
  Index index_;
  Complex accum_;

  Forward& forward_;
  std::vector<std::pair<Index, Complex>> states_;
};

}  // namespace detail

Eigen::VectorXcd LocalValuesV2(Eigen::Ref<const RowMatrix<double>> samples,
                               Eigen::Ref<const Eigen::VectorXcd> values,
                               RbmSpinV2& machine, AbstractOperator& op,
                               Index batch_size) {
  Eigen::VectorXcd locals(samples.rows());
  detail::Forward forward{machine, batch_size};
  detail::Accumulator acc{locals, forward};
  for (auto i = Index{0}; i < samples.rows(); ++i) {
    acc(values(i));
    auto v = Eigen::Ref<const Eigen::VectorXd>{samples.row(i)};
    op.ForEachConn(v, [v, &acc](const ConnectorRef& conn) { acc(v, conn); });
  }
  assert(samples.rows() > 0);
  acc.Finalize(samples.row(0));
  return locals;
}

Eigen::VectorXcd Gradient(Eigen::Ref<const Eigen::VectorXcd> locals,
                          Eigen::Ref<const RowMatrix<Complex>> gradients) {
  if (locals.size() != gradients.rows()) {
    std::ostringstream msg;
    msg << "incompatible dimensions: [" << locals.size() << "] and ["
        << gradients.rows() << ", " << gradients.cols()
        << "]; expected [N] and [N, ?]";
    throw InvalidInputError{msg.str()};
  }
  Eigen::VectorXcd force(gradients.cols());
  Eigen::Map<VectorXcd>{force.data(), force.size()}.noalias() =
      gradients.conjugate() * (locals.array() - locals.mean()).matrix();
  return force;
}

}  // namespace netket
