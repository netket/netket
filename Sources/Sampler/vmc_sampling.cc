#include "Utils/messages.hpp"
#include "vmc_sampling.hpp"

namespace netket {

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

MCResult ComputeSamples(AbstractSampler& sampler, Index num_samples,
                        Index num_skipped,
                        nonstd::optional<std::string> der_logs) {
  NETKET_CHECK(num_samples >= 0, InvalidInputError,
               "invalid number of samples: "
                   << num_samples << "; expected a non-negative integer");
  NETKET_CHECK(num_skipped >= 0, InvalidInputError,
               "invalid number of samples to discard: "
                   << num_skipped << "; expected a non-negative integer");
  NETKET_CHECK(
      !der_logs.has_value() ||
          (*der_logs == "normal" || *der_logs == "centered"),
      InvalidInputError,
      "invalid der_logs: " << *der_logs
                           << "; possible values are 'normal' and 'centered'");
  sampler.Reset();

  const auto num_batches =
      (num_samples + sampler.BatchSize() - 1) / sampler.BatchSize();
  num_samples = num_batches * sampler.BatchSize();
  RowMatrix<double> samples(num_samples, sampler.GetMachine().Nvisible());
  Eigen::VectorXcd values(num_samples);
  auto gradients =
      der_logs.has_value()
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

  if (der_logs.has_value() && *der_logs == "centered")
    detail::SubtractMean(*gradients);
  return {std::move(samples), std::move(values), std::move(gradients),
          sampler.BatchSize()};
}

Eigen::VectorXcd Gradient(Eigen::Ref<const Eigen::VectorXcd> locals,
                          Eigen::Ref<const RowMatrix<Complex>> der_logs) {
  if (locals.size() != der_logs.rows()) {
    std::ostringstream msg;
    msg << "incompatible dimensions: [" << locals.size() << "] and ["
        << der_logs.rows() << ", " << der_logs.cols()
        << "]; expected [N] and [N, ?]";
    throw InvalidInputError{msg.str()};
  }
  Eigen::VectorXcd force(der_logs.cols());
  Eigen::Map<VectorXcd>{force.data(), force.size()}.noalias() =
      der_logs.adjoint() * locals / der_logs.rows();
  MeanOnNodes<>(force);
  return force;
}

}  // namespace netket
