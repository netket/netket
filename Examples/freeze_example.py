"""
Example: layer-wise training by freezing parameters
====================================================

This script shows how to use ``nk.vqs.freeze_parameters`` to optimise only a
subset of a model's parameters, keeping the rest fixed.
"""

import netket as nk

# ── 1. System ──────────────────────────────────────────────────────────────
L = 6
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=3.14)

# ── 2. A two-layer MLP ansatz ──────────────────────────────────────────────
# nk.models.MLP wraps a hidden Dense layer (Dense_0) and an output Dense layer
# (Dense_1).  The parameter tree looks like:
#   {'MLP_0': {'Dense_0': {'kernel', 'bias'},   # first / hidden layer
#              'Dense_1': {'kernel'}}}           # output layer
ma = nk.models.MLP(hidden_dims=(32, 32, 16), param_dtype=float)
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
vs = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=10)

op = nk.optimizer.Sgd(learning_rate=0.05)

# ── 3. Train the full network ──────────────────────────────────────────────
print("\n=== Phase 1: training the full network ===")
gs = nk.driver.VMC(ha, op, variational_state=vs)
gs.run(n_iter=100, out="freeze_phase1")
print(f"Trainable parameters: {nk.jax.tree_size(vs.parameters)}")
print(f"Energy after phase 1: {vs.expect_to_precision(ha, rtol=1e-4)}")

# ── 4. Freeze the first layer ──────────────────────────────────────────────
# The predicate freezes every leaf whose path contains "Dense_0", i.e. the
# whole hidden layer.  Only the output layer (Dense_1) stays trainable.
frozen_vs = nk.vqs.freeze_parameters(vs, lambda path, leaf: "Dense_2" not in path)

print("\n=== Frozen the first (hidden) layer ===")
print(f"Trainable parameters: {nk.jax.tree_size(frozen_vs.parameters)}")
print(
    f"Frozen parameters:    "
    f"{nk.jax.tree_size(frozen_vs.model_state['frozen_params'])}"
)

# ── 5. Continue training only the output layer ─────────────────────────────
print("\n=== Phase 2: training only the output layer ===")
gs2 = nk.driver.VMC(ha, op, variational_state=frozen_vs)
gs2.run(n_iter=100, out="freeze_phase2")
print(f"Energy after phase 2: {frozen_vs.expect_to_precision(ha, rtol=1e-4)}")

# ── 6. Unfreeze (optional) ─────────────────────────────────────────────────
# nk.vqs.unfreeze_parameters restores every parameter to the trainable set,
# recovering a state equivalent to the original (fully trainable) one.
unfrozen_vs = nk.vqs.unfreeze_parameters(frozen_vs)
