import netket as nk

L = 8
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
ma = nk.models.RBM(alpha=10, dtype=float)
vs = nk.vqs.ExactState(hi, ma)

op = nk.optimizer.Sgd(learning_rate=0.003)

gs = nk.driver.VMC(
    ha,
    op,
    variational_state=vs,
    preconditioner=nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianPyTree),
)

log = nk.logging.RuntimeLog()
gs.run(n_iter=5000, out=log)

print(vs.expect(ha))

E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
print("Exact ground state energy = {0:.5f}".format(E_gs[0]))
