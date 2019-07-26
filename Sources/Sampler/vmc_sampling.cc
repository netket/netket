#include <fstream>
#include "Utils/messages.hpp"
#include "vmc_sampling.hpp"

namespace netket {

#if 0
MCResult ComputeSamples(AbstractSampler& sampler, Index nsamples,
                        Index ndiscard, bool compute_logderivs) {
  sampler.Reset();

  for (Index i = 0; i < ndiscard; i++) {
    sampler.Sweep();
  }

  const Index nvisible = sampler.GetMachine().Nvisible();
  const Index npar = sampler.GetMachine().Npar();
  MatrixXd samples(nvisible, nsamples);
  VectorXcd log_vals(nsamples);

  nonstd::optional<MatrixXcd> log_derivs;
  if (compute_logderivs) {
    log_derivs.emplace(npar, nsamples);
    for (Index i = 0; i < nsamples; i++) {
      sampler.Sweep();
      samples.col(i) = VisibleLegacy(sampler);
      log_derivs->col(i) = sampler.GetMachine().DerLogSingle(samples.col(i));
      log_vals(i) = sampler.GetMachine().LogValSingle(samples.col(i));
    }
    // std::ofstream out_file_3{"gradients_2.txt"};
    // out_file_3 << *log_derivs << '\n';

    // Compute "centered" log-derivatives, i.e., O_k ↦ O_k - ⟨O_k⟩
    VectorXcd log_der_mean = log_derivs->rowwise().mean();
    MeanOnNodes<>(log_der_mean);
    log_derivs = log_derivs->colwise() - log_der_mean;
  } else {
    for (Index i = 0; i < nsamples; i++) {
      sampler.Sweep();
      samples.col(i) = VisibleLegacy(sampler);
    }
  }
  return {RowMatrix<double>(std::move(samples).transpose()),
          std::move(log_vals), RowMatrix<Complex>(log_derivs->transpose())};
}
#endif

#if 0
Complex LocalValueLegacy(const AbstractOperator& op, AbstractMachine& psi,
                         Eigen::Ref<const VectorXd> v) {
  AbstractOperator::ConnectorsType tochange;
  AbstractOperator::NewconfsType newconf;
  AbstractOperator::MelType mels;

  op.FindConn(v, mels, tochange, newconf);

  auto logvaldiffs = psi.LogValDiff(v, tochange, newconf);

  assert(mels.size() == std::size_t(logvaldiffs.size()));

  Complex result = 0.0;
  for (Index i = 0; i < logvaldiffs.size(); ++i) {
    result += mels[i] * std::exp(logvaldiffs(i));  // O(v,v') * Ψ(v)/Ψ(v')
  }

  return result;
}
#endif

#if 0
VectorXcd LocalValues(const AbstractOperator& op, AbstractMachine& psi,
                      Eigen::Ref<const MatrixXd> vs) {
  VectorXcd out(vs.cols());
  for (Index i = 0; i < vs.cols(); ++i) {
    out(i) = LocalValue(op, psi, vs.col(i));
  }
  return out;
}
#endif

inline VectorXcd LocalValueDeriv(const AbstractOperator& op,
                                 AbstractMachine& psi,
                                 Eigen::Ref<const VectorXd> v) {
  AbstractOperator::ConnectorsType tochange;
  AbstractOperator::NewconfsType newconf;
  AbstractOperator::MelType mels;
  op.FindConn(v, mels, tochange, newconf);

  auto logvaldiffs = psi.LogValDiff(v, tochange, newconf);
  auto log_deriv = psi.DerLogSingle(v);

  VectorXcd grad(v.size());
  for (int i = 0; i < logvaldiffs.size(); i++) {
    const auto melval = mels[i] * std::exp(logvaldiffs(i));
    const auto log_deriv_prime = psi.DerLogChanged(v, tochange[i], newconf[i]);
    grad += melval * (log_deriv - log_deriv_prime);
  }

  return grad;
}

namespace detail {
class MeanAndVarianceAccumulator {
  nonstd::span<double> const mu_;
  nonstd::span<double> const M2_;
  Index n_;

 public:
  MeanAndVarianceAccumulator(nonstd::span<double> mu, nonstd::span<double> var)
      : mu_{mu}, M2_{var}, n_{0} {
    CheckShape(__FUNCTION__, "var", mu_.size(), M2_.size());
  }

  void operator()(nonstd::span<const double> xs) {
    ++n_;
    const auto kernel = [this](double& mu, double& M2, const double x) {
      const auto delta = x - mu;
      mu += delta / n_;
      M2 += delta * (x - mu);
    };
    for (auto i = Index{0}; i < mu_.size(); ++i) {
      kernel(mu_[i], M2_[i], xs[i]);
    }
  }

  ~MeanAndVarianceAccumulator() {
    if (n_ == 0) {
      std::fill(mu_.begin(), mu_.end(),
                std::numeric_limits<double>::quiet_NaN());
      std::fill(M2_.begin(), M2_.end(),
                std::numeric_limits<double>::quiet_NaN());
    } else if (n_ == 1) {
      std::fill(M2_.begin(), M2_.end(),
                std::numeric_limits<double>::quiet_NaN());
    } else {
      // TODO: Are we __absolutely__ sure we want to divide by `n_` rather than
      // `n_ - 1`?
      const auto scale = 1.0 / static_cast<double>(n_);
      std::for_each(M2_.begin(), M2_.end(), [scale](double& x) { x *= scale; });
    }
  }
};
}  // namespace detail

/// \brief Computes in-chain means and variances.
std::pair<Eigen::VectorXcd, Eigen::VectorXd> StatisticsLocal(
    Eigen::Ref<const Eigen::VectorXcd> values, Index number_chains) {
  NETKET_CHECK(number_chains > 0, InvalidInputError,
               "invalid number of chains: " << number_chains
                                            << "; expected a positive integer");
  NETKET_CHECK(
      values.size() % number_chains == 0, InvalidInputError,
      "invalid number of chains: "
          << number_chains
          << "; `values.size()` must be a multiple of `number_chains`, but "
          << values.size() << " % " << number_chains << " = "
          << values.size() % number_chains);
  Eigen::VectorXcd mean(number_chains);
  // Buffer for variances of real and imaginary parts
  Eigen::VectorXd var(2 * number_chains);
  {
    detail::MeanAndVarianceAccumulator acc{
        nonstd::span<double>{reinterpret_cast<double*>(mean.data()),
                             2 * mean.size()},
        var};
    const auto* p = reinterpret_cast<const double*>(values.data());
    const auto n = values.size() / number_chains;
    for (auto i = Index{0}; i < n; ++i, p += 2 * number_chains) {
      acc({p, 2 * number_chains});
    }
    // Beware: destructor of acc does some work here!!
  }
  for (auto i = Index{0}; i < number_chains; ++i) {
    var(i) = var(2 * i) + var(2 * i + 1);
  }
  var.conservativeResize(number_chains);
  return std::make_pair(std::move(mean), std::move(var));
}

Stats Statistics(Eigen::Ref<const Eigen::VectorXcd> values,
                 Index local_number_chains) {
  NETKET_CHECK(values.size() >= local_number_chains, InvalidInputError,
               "not enough samples to compute statistics");
  constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
  auto stats_local = StatisticsLocal(values, local_number_chains);
  // Number of samples in each Markov Chain
  const auto n = values.size() / local_number_chains;
  // Total number of Markov Chains we have:
  //   #processes x local_number_chains
  const auto m = []() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return static_cast<Index>(size);
  }() * local_number_chains;

  // Calculates the mean over all Markov Chains
  const auto mean = [&stats_local, m]() {
    // Sum over Markov Chains on this MPI node
    Complex local_mean = stats_local.first.sum();
    // Sum over all MPI nodes
    Complex global_mean;
    MPI_Allreduce(&local_mean, &global_mean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
    // Average
    return global_mean / static_cast<double>(m);
  }();

  // (B / n, W)
  const auto var = [&stats_local, m, mean, NaN]() {
    double local_var[2] = {(stats_local.first.array() - mean).abs2().sum(),
                           stats_local.second.array().sum()};
    double global_var[2];
    MPI_Allreduce(&local_var, &global_var, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    assert(m > 0);
    if (m == 1) {  // We can't estimate variance and error is there is only one
                   // chain.
      return std::make_pair(NaN, NaN);
    }
    // TODO: Are the coefficients here correct??
    return std::make_pair(global_var[0] / static_cast<double>(m - 1),
                          global_var[1] / static_cast<double>(m));
  }();

  if (!std::isnan(var.first) && !std::isnan(var.second)) {
    const auto t = var.first / var.second;
    auto correlation = 0.5 * (static_cast<double>(n) * t * t - 1.0);
    if (correlation < 0.0) correlation = NaN;
    const auto R =
        std::sqrt(static_cast<double>(n - 1) / static_cast<double>(n) +
                  var.first / var.second);
    return {mean, std::sqrt(var.first), var.second, correlation, R};
  }
  return Stats{mean, NaN, NaN, NaN, NaN};
}

#if 0
Stats Expectation(const Result& result, AbstractMachine& psi,
                  const AbstractOperator& op) {
  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    bin << loc.real();
  }
  return bin.AllStats();
}

Stats Expectation(const Result& result, AbstractMachine& psi,
                  const AbstractOperator& op, VectorXcd& locvals) {
  locvals.resize(result.NSamples());

  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    locvals(i) = loc;
    bin << loc.real();
  }

  return bin.AllStats();
}
#endif

#if 0
Stats Variance(const Result& result, AbstractMachine& psi,
               const AbstractOperator& op) {
  VectorXcd locvals;
  auto ex = Expectation(result, psi, op, locvals);
  return Variance(result, psi, op, ex.mean, locvals);
}

Stats Variance(const Result& /*result*/, AbstractMachine& /*psi*/,
               const AbstractOperator& /*op*/, double expectation_value,
               const VectorXcd& locvals) {
  Binning<double> bin_var;
  for (Index i = 0; i < locvals.size(); ++i) {
    bin_var << std::norm(locvals(i) - expectation_value);
  }
  return bin_var.AllStats();
}
#endif

#if 0
VectorXcd Gradient(const Result& result, AbstractMachine& psi,
                   const AbstractOperator& op) {
  if (!result.LogDerivs().has_value()) {
    throw std::runtime_error{
        "vmc::Result does not contain log-derivatives, which are required to "
        "compute gradients."};
  }

  VectorXcd locvals;
  Expectation(result, psi, op, locvals);

  VectorXcd grad =
      result.LogDerivs()->conjugate() * locvals / double(result.NSamples());
  MeanOnNodes<>(grad);

  return grad;
}
#endif

#if 0
VectorXcd Gradient(const Result& result, AbstractMachine& /*psi*/,
                   const AbstractOperator& /*op*/, const VectorXcd& locvals) {
  if (!result.LogDerivs().has_value()) {
    throw std::runtime_error{
        "vmc::Result does not contain log-derivatives, which are required to "
        "compute gradients."};
  }

  VectorXcd grad =
      result.LogDerivs()->conjugate() * locvals / double(result.NSamples());
  MeanOnNodes<>(grad);
  return grad;
}
#endif

VectorXcd GradientOfVariance(Eigen::Ref<const RowMatrix<double>> samples,
                             Eigen::Ref<const Eigen::VectorXcd> local_values,
                             AbstractMachine& psi, const AbstractOperator& op) {
  CheckShape(__FUNCTION__, "samples", {samples.rows(), samples.cols()},
             {std::ignore, psi.Nvisible()});
  CheckShape(__FUNCTION__, "local_values", local_values.size(), samples.rows());
  // TODO: This function can probably be implemented more efficiently (e.g.,
  // by computing local values and their gradients at the same time or reusing
  // already computed local values)
  RowMatrix<Complex> locval_deriv(samples.rows(), psi.Npar());
  for (auto i = Index{0}; i < samples.rows(); ++i) {
    locval_deriv.row(i) =
        LocalValueDeriv(op, psi, samples.row(i).transpose()).transpose();
  }
  VectorXcd locval_deriv_mean = locval_deriv.colwise().mean();
  MeanOnNodes<>(locval_deriv_mean);
  locval_deriv = locval_deriv.colwise() - locval_deriv_mean;

  VectorXcd grad = locval_deriv.conjugate() * local_values /
                   static_cast<double>(samples.rows());
  MeanOnNodes<>(grad);
  return grad;
}

namespace detail {
void SubtractMean(RowMatrix<Complex>& gradients) {
  VectorXcd mean = gradients.colwise().mean();
  assert(mean.size() == gradients.cols());
  MeanOnNodes<>(mean);
  gradients.rowwise() -= mean.transpose();
}
}  // namespace detail

#if 1
MCResult ComputeSamples(AbstractSampler& sampler, Index num_samples,
                        Index num_skipped, bool compute_gradients) {
  NETKET_CHECK(num_samples >= 0, InvalidInputError,
               "invalid number of samples: "
                   << num_samples << "; expected a non-negative integer");
  NETKET_CHECK(num_skipped >= 0, InvalidInputError,
               "invalid number of samples to discard: "
                   << num_skipped << "; expected a non-negative integer");
  sampler.Reset();

  const auto num_batches =
      (num_samples + sampler.BatchSize() - 1) / sampler.BatchSize();
  num_samples = num_batches * sampler.BatchSize();
  RowMatrix<double> samples(num_samples, sampler.GetMachine().Nvisible());
  Eigen::VectorXcd values(num_samples);
  auto gradients =
      compute_gradients
          ? nonstd::optional<RowMatrix<Complex>>{nonstd::in_place, num_samples,
                                                 sampler.GetMachine().Npar()}
          : nonstd::nullopt;

  struct Record {
    AbstractSampler& sampler_;
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
      sampler_.GetMachine().DerLog(X, out, any{});
    }

    void operator()() {
      assert(i_ * sampler_.BatchSize() < samples_.rows());
      Batch() = sampler_.CurrentState();
      if (gradients_.has_value()) Gradients();
      ++i_;
    }
  } record{sampler, samples, values, gradients, 0};

  for (auto i = Index{0}; i < num_skipped; ++i) {
    sampler.Sweep();
  }
  if (num_batches > 0) {
    record();
    for (auto i = Index{1}; i < num_batches; ++i) {
      sampler.Sweep();
      record();
    }
  }

  if (gradients.has_value()) detail::SubtractMean(*gradients);
  return {std::move(samples), std::move(values), std::move(gradients)};
}
#endif

namespace detail {
/// A helper class for forward propagation of batches through machines.
struct Forward {
  Forward(AbstractMachine& m, Index batch_size)
      : machine_{m},
        X_(batch_size, m.Nvisible()),
        Y_(batch_size),
        coeff_(batch_size),
        i_{0} {}

  /// \brief Returns whether internal buffer if full.
  bool Full() const noexcept { return i_ == BatchSize(); }
  /// \brief Returns whether internal buffer if empty.
  bool Empty() const noexcept { return i_ == 0; }
  Index BatchSize() const noexcept { return Y_.size(); }

  /// \brief Add an element to internal buffer.
  ///
  /// Buffer should not be full!
  void Push(Eigen::Ref<const Eigen::VectorXd> v, const ConnectorRef& conn) {
    assert(!Full());
    X_.row(i_) = v.transpose();
    for (auto j = Index{0}; j < conn.tochange.size(); ++j) {
      X_(i_, conn.tochange[j]) = conn.newconf[j];
    }
    coeff_(i_) = conn.mel;
    ++i_;
  }

  /// \brief Fills the remaining part of internal buffer with visible
  /// configuration \p v.
  void Fill(Eigen::Ref<const Eigen::VectorXd> v) {
    assert(!Empty() && !Full());
    const auto n = BatchSize() - i_;
    X_.block(i_, 0, n, X_.cols()) = v.transpose().colwise().replicate(n);
    coeff_.segment(i_, n).setConstant(0.0);
    i_ += n;
    assert(Full());
  }

  /// \brief Runs forward propagation.
  ///
  /// Buffer should be full!
  std::tuple<const Eigen::VectorXcd&, Eigen::VectorXcd&> Propagate() {
    assert(Full());
    machine_.LogVal(X_, /*out=*/Y_, /*cache=*/any{});
    i_ = 0;
    return std::tuple<const Eigen::VectorXcd&, Eigen::VectorXcd&>{coeff_, Y_};
  }

 private:
  AbstractMachine& machine_;
  RowMatrix<double> X_;
  Eigen::VectorXcd Y_;
  Eigen::VectorXcd coeff_;
  Index i_;
};

struct Accumulator {
  Accumulator(Eigen::VectorXcd& loc, Forward& fwd)
      : locals_{loc}, index_{0}, accum_{0.0, 0.0}, forward_{fwd}, states_{} {
    states_.reserve(forward_.BatchSize());
  }

  void operator()(Eigen::Ref<const Eigen::VectorXd> v,
                  const ConnectorRef& conn) {
    assert(!forward_.Full());
    forward_.Push(v, conn);
    ++states_.back().first;
    if (forward_.Full()) {
      ProcessBatch();
    }
    assert(!forward_.Full());
  }

  void operator()(Complex log_val) {
    assert(!forward_.Full());
    states_.emplace_back(0, log_val);
  }

  /// Number of visible configurations we've processed is not necessarily a
  /// multiple of batch size.
  void Finalize(Eigen::Ref<const Eigen::VectorXd> v) {
    if (forward_.Empty()) {
      locals_(index_) = accum_;
      // ++index_;
      // accum_ = 0.0;
      return;
    }
    states_.emplace_back(0, states_.back().second);
    forward_.Fill(v);
    ProcessBatch();
  }

 private:
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
      // We do the last iteration separately, because we don't yet know the
      // number of spin configurations which contribute to the last local energy
      // (some of them might not fit into current batch).
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
      // Computes local value
      const auto& state = states_[j];
      for (auto n = Index{0}; n < state.first; ++n, ++i) {
        accum_ += y(i) * coeff(i);
      }
      // Stores it and resets the accumulator
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
    // A fancy way to throw away all elements except for the first one
    if (states_.size() > 1) {
      std::swap(states_.front(), states_.back());
      states_.resize(1);
    }
    states_.front().first = 0;
  }

  Eigen::VectorXcd& locals_;  // Destination array
  Index index_;               // Index in locals_
  Complex accum_;             // Accumulator for current local energy

  Forward& forward_;
  // A priori it is unknown whether H|v⟩ contains more basis vectors than can
  // fit into a batch. If H|v⟩ contains fewer than batch size basis vectors,
  // then during one forward propagation, we will be computing log(ψ(v')) for v'
  // which contribute to different local energies. `states_` vector keeps track
  // of all local energies we're currently computing. Each state is a pair of:
  //   * number of v' which contribute to ⟨v|H|ψ⟩/⟨v|ψ⟩ and value log(⟨v|ψ⟩).
  std::vector<std::pair<Index, Complex>> states_;
};

}  // namespace detail

Eigen::VectorXcd LocalValues(Eigen::Ref<const RowMatrix<double>> samples,
                             Eigen::Ref<const Eigen::VectorXcd> values,
                             AbstractMachine& machine,
                             const AbstractOperator& op, Index batch_size) {
  if (batch_size < 1) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size << "; expected >=1";
    throw InvalidInputError{msg.str()};
  }
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
      gradients.adjoint() * locals / gradients.rows();
  MeanOnNodes<>(force);
  return force;
}

}  // namespace netket
