
import jax

jax.config.update("jax_cpu_enable_gloo_collectives", True)
jax.distributed.initialize()

import numpy as np
import netket as nk


def test_fullsumstate(chunk_size):
    L = 8
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ma = nk.models.RBM(alpha=1, param_dtype=np.complex128)
    vs = nk.vqs.FullSumState(hi, ma, chunk_size=chunk_size)
    ha = nk.operator.IsingJax(hilbert=vs.hilbert, graph=g, h=1.0)
    opt = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(
        nk.optimizer.qgt.QGTOnTheFly(holomorphic=True), diag_shift=0.01
    )
    gs = nk.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    gs.run(5)

test_fullsumstate(64)
