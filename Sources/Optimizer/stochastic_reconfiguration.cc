#include "stochastic_reconfiguration.hpp"

namespace netket {

nonstd::optional<SR::LSQSolver> SR::SolverFromString(const std::string& name) {
  if (name == "LLT") {
    return LLT;
  } else if (name == "LDLT") {
    return LDLT;
  } else if (name == "ColPivHouseholder") {
    return ColPivHouseholder;
  } else if (name == "BDCSVD") {
    return BDCSVD;
  } else {
    return nonstd::nullopt;
  }
}

const char* SR::SolverAsString(LSQSolver solver) {
  static const char* solvers[] = {"LLT", "LDLT", "ColPivHouseholder", "BCDSVD"};
  return solvers[solver];
}

void SR::ComputeUpdate(OkRef Oks, GradRef grad_ref, OutputRef deltaP) {
  double nsamp = Oks.rows();
  SumOnNodes(nsamp);
  // auto npar = grad.size();

  // TODO: Is this copy avoidable?
  VectorXcd grad = grad_ref;

  if (is_holomorphic_) {
    if (use_iterative_) {
      SolveIterative<VectorXcd>(Oks, grad, deltaP, nsamp);
    } else {
      BuildSMatrix<MatrixXcd>(Oks.adjoint() * Oks, Scomplex_, nsamp);
      ApplyPreconditioning(Scomplex_, grad);
      SolveLeastSquares<MatrixXcd, VectorXcd>(Scomplex_, grad, deltaP);
      RevertPreconditioning(deltaP);
    }
  } else {
    if (use_iterative_) {
      SolveIterative<VectorXd>(Oks, grad.real(), deltaP, nsamp);
    } else {
      BuildSMatrix<MatrixXd>((Oks.adjoint() * Oks).real(), Sreal_, nsamp);
      ApplyPreconditioning(Sreal_, grad);
      SolveLeastSquares<MatrixXd, VectorXd>(Sreal_, grad.real(), deltaP.real());
      RevertPreconditioning(deltaP);
    }
    deltaP.imag().setZero();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void SR::SetParameters(LSQSolver solver, double diagshift, bool use_iterative,
                       bool is_holomorphic) {
  CheckSolverCompatibility(use_iterative, solver, store_rank_);

  solver_ = solver;
  sr_diag_shift_ = diagshift;
  use_iterative_ = use_iterative;
  is_holomorphic_ = is_holomorphic;
}

void SR::SetParameters(double diagshift, bool use_iterative, bool use_cholesky,
                       bool is_holomorphic) {
  SetParameters(use_cholesky ? LLT : ColPivHouseholder, diagshift,
                use_iterative, is_holomorphic);
}

std::string SR::LongDesc(Index depth) const {
  auto indent = [&depth]() { return std::string(4 * depth, ' '); };
  std::ostringstream str;
  str << indent() << "Stochastic reconfiguration method for "
      << (is_holomorphic_ ? "holomorphic" : "real-parameter")
      << " wavefunctions\n"
      << indent() << "Solver: ";
  if (use_iterative_) {
    str << "iterative (Conjugate Gradient)";
  } else {
    str << SolverAsString(solver_);
  }
  str << "\n";
  return str.str();
}

std::string SR::ShortDesc() const {
  std::stringstream str;
  str << "SR(solver=";
  if (use_iterative_) {
    str << "iterative";
  } else {
    str << SolverAsString(solver_) << ", diag_shift=" << sr_diag_shift_;
  }
  str << ", is_holomorphic=" << (is_holomorphic_ ? "True" : "False") << ")";
  return str.str();
}

void SR::SetStoreRank(bool enabled) {
  CheckSolverCompatibility(use_iterative_, solver_, enabled);
  store_rank_ = enabled;
  if (!enabled) {
    last_rank_ = nonstd::nullopt;
  }
}

void SR::SetStoreFullSMatrix(bool enabled) {
  if (use_iterative_ && enabled) {
    throw std::logic_error{
        "Cannot store full S matrix with `use_iterative = true`."};
  }
  store_full_S_matrix_ = enabled;
  if (!enabled) {
    last_S_ = nonstd::nullopt;
  }
}
}  // namespace netket
