import jax
import jax.numpy as jnp
from netket.sampler.rules import ExchangeRule


class FermionExchangeRule(ExchangeRule):
    """Exchange rule for fermions.

    Works similarly to exchange rule, but:
    takes into account that only occupied orbitals
    can be exchanged with unoccupied ones.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]

        hoppable_clusters = _compute_hoppable_clusters_mask(rule.clusters, σ)
        # compute a mask for the clusters that can be hopped

        keys = jnp.asarray(jax.random.split(key, n_chains))

        def _update_sample(key, σ, hoppable_clusters):
            # pick a random cluster, taking into account the mask
            n_conn = hoppable_clusters.sum(axis=-1)
            cluster = jax.random.choice(
                key,
                a=jnp.arange(rule.clusters.shape[0]),
                p=hoppable_clusters,
                replace=True,
            )

            # sites to be exchanged
            si = rule.clusters[cluster, 0]
            sj = rule.clusters[cluster, 1]

            σp = σ.at[si].set(σ[sj])
            σp = σp.at[sj].set(σ[si])

            # compute the number of connected sites
            hoppable_clusters_proposed = _compute_hoppable_clusters_mask(
                rule.clusters, σp
            )
            n_conn_proposed = hoppable_clusters_proposed.sum(axis=-1)
            log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)
            return σp, log_prob_corr

        return jax.vmap(_update_sample, in_axes=(0, 0, 0), out_axes=0)(
            keys, σ, hoppable_clusters
        )

    def __repr__(self):
        return f"FermionExchangeRule(# of clusters: {len(self.clusters)})"


@jax.jit
def _compute_hoppable_clusters_mask(clusters, σ):
    # see which modes are occupied
    occ = jnp.isclose(σ, 1)
    # mask the clusters to include only feasible moves (occ -> unocc, or the inverse)
    hoppable_clusters_mask = jnp.logical_xor(
        occ[..., clusters[:, 0]], occ[..., clusters[:, 1]]
    )
    return hoppable_clusters_mask
