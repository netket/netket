"""
Example: thermalising Markov chains with MCState.thermalise
===========================================================

This script shows when and how to use ``vs.thermalise(op)`` to advance
Markov chains until they are well-mixed before making final measurements.

The scenario
------------
After optimisation the variational state parameters are fixed, but the
sampler chains may still be stuck in a corner of Hilbert space — especially
if the last optimisation step moved the wavefunction significantly or if the
chains were initialised from a biased configuration.  In that case the first
``vs.expect(ha)`` call gives an unreliable estimate whose bias is invisible
unless you check R_hat.

``vs.thermalise(op)`` runs the sampler (at the current sweep_size) and
monitors the Gelman-Rubin R_hat with a short EMA window, stopping as soon as
R_hat < rhat_tol has been sustained for ``patience`` consecutive iterations.
Unlike ``check_mc_convergence`` it mutates the state in-place, so the chains
are left at a well-mixed position ready for measurements.
"""

import numpy as np
import netket as nk
import matplotlib.pyplot as plt

# ── 1. System ──────────────────────────────────────────────────────────────
L = 16
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# ── 2. Model + sampler ─────────────────────────────────────────────────────
ma = nk.models.RBM(alpha=1, param_dtype=float)
sa = nk.sampler.MetropolisLocal(hi, n_chains=64, sweep_size=L)
vs = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=0)

# ── 3. Run optimisation ────────────────────────────────────────────────────
import optax

gs = nk.driver.VMC_SR(ha, optax.sgd(0.02), variational_state=vs, diag_shift=0.01)
gs.run(n_iter=300)

print("\n=== State after optimisation ===")
e = vs.expect(ha)
print(f"Energy (unthermalized): {e}")

# ── 4. Manually bias the chains (simulate a bad start) ────────────────────
# Force all chains to the all-spins-up configuration to create a
# deliberately poor starting point.  In practice this can happen after
# loading a checkpoint or after a large parameter update.
all_up = np.ones((vs.sampler.n_chains, hi.size), dtype=np.int8)
vs.sampler_state = vs.sampler_state.replace(σ=all_up)
print("\nChains forcibly reset to all-up — R_hat will be large initially.")

# ── 5. Thermalise ──────────────────────────────────────────────────────────
# Default parameters work well for most problems:
#   rhat_tol=1.05 — declare converged when R_hat < 1.05
#   patience=10   — require 10 consecutive good checks
#   decay=0.9     — EMA window of ~10 batches
stats, hist = vs.thermalise(
    ha,
    min_chain_length=50,
    max_chain_length=2000,
    rhat_tol=1.05,
    patience=10,
    verbose=True,
)

print("\n=== After thermalisation ===")
e = vs.expect(ha)
print(f"Energy (thermalized):   {e}")
print(f"Final R_hat:            {stats.R_hat:.4f}")

# ── 6. Plot R_hat evolution ────────────────────────────────────────────────
rhats = hist["R_hat"].values
steps = hist["R_hat"].iters  # cumulative samples/chain

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(steps, rhats, lw=1.5, color="C0", label=r"$\hat{R}$ (EMA)")
ax.axhline(1.05, color="gold", lw=1.2, ls="--", label=r"$\hat{R}=1.05$ (target)")
ax.axhline(1.10, color="salmon", lw=1.2, ls="--", label=r"$\hat{R}=1.10$ (bad)")
ax.set_xlabel("Cumulative samples / chain")
ax.set_ylabel(r"$\hat{R}$")
ax.set_title("R_hat during thermalisation")
ax.legend(fontsize="small")
fig.tight_layout()
plt.savefig("thermalise_rhat.png", dpi=150)
plt.show()

# ── 7. Verify with check_mc_convergence ───────────────────────────────────
# Now that chains are thermalized, check_mc_convergence can also confirm
# the sweep_size is sufficient for decorrelated samples.
print("\n=== Verifying with check_mc_convergence ===")
conv_stats, conv_hist = vs.check_mc_convergence(ha, min_chain_length=200, plot=True)
