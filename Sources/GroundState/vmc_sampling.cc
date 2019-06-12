#include "vmc_sampling.hpp"

namespace netket {
namespace vmc {

Result PerformSampling(AbstractSampler &sampler, Index nsamples,
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

void LocalValues(const AbstractOperator &op, AbstractMachine &psi,
                 Eigen::Ref<const MatrixXd> vs, VectorXcd &out) {
  out.resize(vs.cols());
  for (Index i = 0; i < vs.cols(); ++i) {
    out(i) = LocalValue(op, psi, vs.col(i));
  }
}

Stats Expectation(Result result, const AbstractOperator &op,
                  AbstractMachine &psi) {
  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    bin << loc.real();
  }
  return bin.AllStats();
}

Stats Expectation(const Result &result, const AbstractOperator &op,
                  AbstractMachine &psi, VectorXcd &locvals) {
  locvals.resize(result.NSamples());

  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    locvals(i) = loc;
    bin << loc.real();
  }

  return bin.AllStats();
}

ExpectationVarianceResult ExpectationVariance(const Result &result,
                                              const AbstractOperator &op,
                                              AbstractMachine &psi,
                                              VectorXcd &locvals) {
  Binning<double> bin_var;

  auto ex_stats = Expectation(result, op, psi, locvals);

  Complex loc_mean = locvals.mean();
  MeanOnNodes<>(loc_mean);
  locvals.array() -= loc_mean;

  // TODO: Mean and variance should be computed in one pass
  for (Index i = 0; i < locvals.size(); ++i) {
    bin_var << std::norm(locvals(i));
  }

  return {ex_stats, bin_var.AllStats()};
}

ExpectationVarianceResult ExpectationVariance(const Result &result,
                                              const AbstractOperator &op,
                                              AbstractMachine &psi) {
  VectorXcd locvals;
  return ExpectationVariance(result, op, psi, locvals);
}

ExpectationVarianceResult ExpectationVarianceGradient(
    const Result &result, const AbstractOperator &op, AbstractMachine &psi,
    VectorXcd &grad) {
  VectorXcd locvals;
  const auto stats = ExpectationVariance(result, op, psi, locvals);

  grad = result.LogDerivs().conjugate() * locvals / double(result.NSamples());
  MeanOnNodes<>(grad);

  return stats;
}

}  // namespace vmc
}  // namespace netket
