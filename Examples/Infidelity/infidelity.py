import netket as nk
import netket.experimental as nkx
import matplotlib.pyplot as plt

# Hilbert space and Hamiltonian
g = nk.graph.Square(4)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
H = nk.operator.Ising(hi, graph=g, h=1.0)

# Target and variational states
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
ma = nk.models.RBM(alpha=1)
vs_target = nk.vqs.MCState(sampler=sa, model=ma, n_samples=1024)
vs = nk.vqs.MCState(sampler=sa, model=ma, n_samples=1024)

# The infidelity operator, that can be used to compute the infidelity between Psi and HPhi
I_op = nkx.observable.InfidelityOperator(vs_target, operator=H)
print("Infidelity among Psi and HPhi: ", vs.expect(I_op))

# Create the driver to minimize the infidelity between Psi and HPhi
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
driver = nkx.driver.Infidelity_SR(
    vs_target,
    optimizer,
    operator=H,
    variational_state=vs,
    diag_shift=0.0001,
)

log = nk.logging.RuntimeLog()

driver.run(
    1000,
    out=log,
)

# Final infidelity
print("Final infidelity among Psi and HPhi: ", vs.expect(I_op))

# Plotting the learning curve
plt.ion()
plt.errorbar(
    log.data["Infidelity"].iters,
    log.data["Infidelity"]["Mean"],
    log.data["Infidelity"]["Sigma"],
)
plt.yscale("log")
plt.show()
