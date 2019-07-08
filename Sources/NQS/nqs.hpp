#include "Machine/rbm_spin.hpp"
#include "Machine/abstract_machine.hpp"
#include "Optimizer/abstract_optimizer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include <vector>



namespace netket {

class NQS {

    RbmSpin psi;
    AbstractSampler sa;
    AbstractOptimizer op;

    public:
        NQS(int nqubits);

        void applyHadamard(int qubit);
        void applyPauliX(int qubit);
        void applyPauliY(int qubit);
        void applyPauliZ(int qubit);
        void applySingleZRotation(int qubit, double theta);
        void applyControlledZRotation(int controlQubit, int qubit, double theta);
        void sample();

    private:

        using VectorType = AbstractMachine::VectorType;
        using MatrixType = AbstractMachine::MatrixType;

        VectorType getPsi_a();
        VectorType getPsi_b();
        MatrixType getPsi_W();
        void setPsiParams(VectorType& a, VectorType& b, MatrixType& W);
    
    };

}  // namespace netket