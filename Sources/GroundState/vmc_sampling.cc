#include "vmc_sampling.hpp"

namespace netket {
namespace vmc {

Result ComputeSamples(AbstractSampler &sampler, Index nsamples,
                      Index ndiscard) {
  sampler.Reset();

  for (Index i = 0; i < ndiscard; i++) {
    sampler.Sweep();
  }

  const Index nvisible = sampler.GetMachine().Nvisible();
  const Index npar = sampler.GetMachine().Npar();
  MatrixXd samples(nvisible, nsamples);
  MatrixXcd log_derivs(npar, nsamples);

  for (Index i = 0; i < nsamples; i++) {
    sampler.Sweep();
    samples.col(i) = sampler.Visible();
    log_derivs.col(i) = sampler.DerLogVisible();
  }

  // Compute "centered" log-derivatives, i.e., O_k ↦ O_k - ⟨O_k⟩
  VectorXcd log_der_mean = log_derivs.rowwise().mean();
  MeanOnNodes<>(log_der_mean);
  log_derivs = log_derivs.colwise() - log_der_mean;

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

void GradientOfVariance(const Result &result, const AbstractOperator &op,
                        AbstractMachine &psi, VectorXcd &grad) {
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

  grad = locval_deriv.conjugate() *
         LocalValues(op, psi, result.SampleMatrix()) /
         double(result.NSamples());
  MeanOnNodes<>(grad);
}

Stats Ex(const Result &result, const AbstractOperator &op,
         AbstractMachine &psi) {
  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    bin << loc.real();
  }
  return bin.AllStats();
}

Stats Ex(const Result &result, const AbstractOperator &op, AbstractMachine &psi,
         VectorXcd &locvals) {
  locvals.resize(result.NSamples());

  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    locvals(i) = loc;
    bin << loc.real();
  }

  return bin.AllStats();
}

ExpectationVarianceResult ExVar(const Result &result,
                                const AbstractOperator &op,
                                AbstractMachine &psi, VectorXcd &locvals) {
  Binning<double> bin_var;

  auto ex_stats = Ex(result, op, psi, locvals);

  Complex loc_mean = locvals.mean();
  MeanOnNodes<>(loc_mean);
  locvals.array() -= loc_mean;

  // TODO: Mean and variance should be computed in one pass
  for (Index i = 0; i < locvals.size(); ++i) {
    bin_var << std::norm(locvals(i));
  }

  return {ex_stats, bin_var.AllStats()};
}

ExpectationVarianceResult ExVar(const Result &result,
                                const AbstractOperator &op,
                                AbstractMachine &psi) {
  VectorXcd locvals;
  return ExVar(result, op, psi, locvals);
}

ExpectationVarianceResult ExVarGrad(const Result &result,
                                    const AbstractOperator &op,
                                    AbstractMachine &psi, VectorXcd &grad) {
  VectorXcd locvals;
  const auto stats = ExVar(result, op, psi, locvals);

  grad = result.LogDerivs().conjugate() * locvals / double(result.NSamples());
  MeanOnNodes<>(grad);

  return stats;
}

}  // namespace vmc
}  // namespace netket
