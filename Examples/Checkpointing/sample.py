from etils import epath
import orbax.checkpoint as ocp

import netket as nk
from netket import experimental as nkx
import optax

# Keeps a maximum of 3 checkpoints, and only saves every other step.
path = epath.Path("/tmp/nkcheckpt-04")

checkpointer = ocp.CheckpointManager(
    path,
    ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
    options=ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1),
)

L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
ma = nk.models.RBM(alpha=1, param_dtype=float)
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))
sr = nk.optimizer.SR(diag_shift=0.01)
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr, checkpointer=checkpointer)
# maybe restore checkpoint
if checkpointer.latest_step() is not None:
    gs.restore_checkpoint()

log = nkx.logging.HDF5Log("log2", mode="append")
log = nk.logging.RuntimeLog()

# Run the optimization for 500 iterations
gs.run(n_iter=49, out=log, show_progress=False)
