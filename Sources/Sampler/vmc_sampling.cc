#include "vmc_sampling.hpp"

namespace netket {
namespace vmc {

Result ComputeSamples(AbstractSampler &sampler, Index nsamples, Index ndiscard,
                      bool compute_logderivs) {
  sampler.Reset();

  for (Index i = 0; i < ndiscard; i++) {
    sampler.Sweep();
  }

  const Index nvisible = sampler.GetMachine().Nvisible();
  const Index npar = sampler.GetMachine().Npar();
  MatrixXd samples(nvisible, nsamples);

  nonstd::optional<MatrixXcd> log_derivs;
  if (compute_logderivs) {
    log_derivs.emplace(npar, nsamples);
    for (Index i = 0; i < nsamples; i++) {
      sampler.Sweep();
      samples.col(i) = sampler.Visible();
      log_derivs->col(i) = sampler.DerLogVisible();
    }

    // Compute "centered" log-derivatives, i.e., O_k ↦ O_k - ⟨O_k⟩
    VectorXcd log_der_mean = log_derivs->rowwise().mean();
    MeanOnNodes<>(log_der_mean);
    log_derivs = log_derivs->colwise() - log_der_mean;
  } else {
    for (Index i = 0; i < nsamples; i++) {
      sampler.Sweep();
      samples.col(i) = sampler.Visible();
    }
  }
  return Result(std::move(samples), std::move(log_derivs));
}

Complex LocalValue(const AbstractOperator &op, AbstractMachine &psi,
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

VectorXcd LocalValues(const AbstractOperator &op, AbstractMachine &psi,
                      Eigen::Ref<const MatrixXd> vs) {
  VectorXcd out(vs.cols());
  for (Index i = 0; i < vs.cols(); ++i) {
    out(i) = LocalValue(op, psi, vs.col(i));
  }
  return out;
}

VectorXcd LocalValueDeriv(const AbstractOperator &op, AbstractMachine &psi,
                          Eigen::Ref<const VectorXd> v) {
  AbstractOperator::ConnectorsType tochange;
  AbstractOperator::NewconfsType newconf;
  AbstractOperator::MelType mels;
  op.FindConn(v, mels, tochange, newconf);

  auto logvaldiffs = psi.LogValDiff(v, tochange, newconf);
  auto log_deriv = psi.DerLog(v);

  VectorXcd grad(v.size());
  for (int i = 0; i < logvaldiffs.size(); i++) {
    const auto melval = mels[i] * std::exp(logvaldiffs(i));
    const auto log_deriv_prime = psi.DerLogChanged(v, tochange[i], newconf[i]);
    grad += melval * (log_deriv - log_deriv_prime);
  }

  return grad;
}

Stats Expectation(const Result &result, AbstractMachine &psi,
                  const AbstractOperator &op) {
  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    bin << loc.real();
  }
  return bin.AllStats();
}

Stats Expectation(const Result &result, AbstractMachine &psi,
                  const AbstractOperator &op, VectorXcd &locvals) {
  locvals.resize(result.NSamples());

  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    locvals(i) = loc;
    bin << loc.real();
  }

  return bin.AllStats();
}

Stats Variance(const Result &result, AbstractMachine &psi,
               const AbstractOperator &op) {
  VectorXcd locvals;
  auto ex = Expectation(result, psi, op, locvals);
  return Variance(result, psi, op, ex.mean, locvals);
}

Stats Variance(const Result & /*result*/, AbstractMachine & /*psi*/,
               const AbstractOperator & /*op*/, double expectation_value,
               const VectorXcd &locvals) {
  Binning<double> bin_var;
  for (Index i = 0; i < locvals.size(); ++i) {
    bin_var << std::norm(locvals(i) - expectation_value);
  }
  return bin_var.AllStats();
}

VectorXcd Gradient(const Result &result, AbstractMachine &psi,
                   const AbstractOperator &op) {
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

VectorXcd Gradient(const Result &result, AbstractMachine & /*psi*/,
                   const AbstractOperator & /*op*/, const VectorXcd &locvals) {
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

VectorXcd GradientOfVariance(const Result &result, AbstractMachine &psi,
                             const AbstractOperator &op) {
  // TODO: This function can probably be implemented more efficiently (e.g., by
  // computing local values and their gradients at the same time or reusing
  // already computed local values)
  MatrixXcd locval_deriv(psi.Npar(), result.NSamples());

  for (int i = 0; i < result.NSamples(); i++) {
    locval_deriv.col(i) = LocalValueDeriv(op, psi, result.Sample(i));
  }
  VectorXcd locval_deriv_mean = locval_deriv.colwise().mean();
  MeanOnNodes<>(locval_deriv_mean);
  locval_deriv = locval_deriv.colwise() - locval_deriv_mean;

  VectorXcd grad = locval_deriv.conjugate() *
                   LocalValues(op, psi, result.SampleMatrix()) /
                   double(result.NSamples());
  MeanOnNodes<>(grad);
  return grad;
}

}  // namespace vmc
}  // namespace netket
