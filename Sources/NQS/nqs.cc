
#include "nqs.hpp"
#include "Graph/hypercube.hpp"
#include "Hilbert/spins.hpp"
#include "Optimizer/ada_max.hpp"

namespace netket {

    NQS::NQS(int nqubits) {
        g = Hypercube(nqubits, 1, false);
        hi = Spin(g, 0.5);
        psi(RbmSpin(hi));
        op(Adamax());
    }

    void NQS::applyHadamard(int qubit) {
        ;
    }

    void NQS::applyPauliX(int qubit) {
        ;
    }

    void NQS::applyPauliY(int qubit) {
        ;
    }

    void NQS::applyPauliZ(int qubit) {
        ;
    }

    void NQS::applySingleZRotation(int qubit, double theta) {
        ;
    }

    void NQS::applyControlledZRotation(int controlQubit, int qubit, double theta) {
        ;
    }

    void NQS::sample() {
        ;
    }



    VectorType NQS::getPsi_a() {
        pars = psi.GetParameters();
        //always "use_a" & "use_b"
        return pars.head(psi.Nvisible());
    }

    VectorType NQS::getPsi_b() {
        pars = psi.GetParameters();
        //always "use_a" & "use_b"
        return pars.segment(psi.Nhidden());
    }

    MatrixType NQS::getPsi_W() {
        pars = psi.GetParameters();
        VectorType Wpars = pars.tail(psi.Nvisible() * psi.Nhidden());
        return Eigen::Map<MatrixType>(Wpars.data(), psi.Nvisible(), psi.Nhidden());
    }

    void NQS::setPsiParams(VectorType& a, VectorType& b, MatrixType& W) {
        VectorType pars(a.size() + b.size() + W.size());

        pars.head(psi.Nvisible()) = a_;
        pars.segment(usea_ * psi.Nvisible(), psi.Nhidden()) = b_;
        pars.tail(psi.Nvisible() * psi.Nhidden()) = Eigen::Map<VectorType>(W_.data(), psi.Nvisible() * psi.Nhidden());

        psi.setParameters(pars);
    }

}  // namespace netket