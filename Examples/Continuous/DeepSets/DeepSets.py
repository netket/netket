import netket as nk
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp


def minimum_distance(x, sdim):

    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)
    # L = 4.818188977945568

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0
    # basis = jnp.array(jnp.meshgrid([0, L, -L])).T.reshape(-1,1)#, [0, L, -L], [0, L, -L])).T.reshape(-1, 3)
    # distances = (distances[:, jnp.newaxis, :] + basis).reshape(-1, sdim)
    return jnp.linalg.norm(distances, axis=1)


def potential(x):
    # compute potential (n_samples, n_chains)
    dis = minimum_distance(x, 1)
    eps = 7.8463738763900351690745308069267072908970259541066716242385224863
    A = 0.544850 * 10 ** 6
    alpha = 13.353384
    c6 = 1.37332412
    c8 = 0.4253785
    c10 = 0.178100
    D = 1.241314

    return jnp.sum(
        eps
        * (
            A * jnp.exp(-alpha * dis)
            - (c6 / dis ** 6 + c8 / dis ** 8 + c10 / dis ** 10)
            * jnp.where(dis < D, jnp.exp(-((D / dis - 1) ** 2)), 1.0)
        )
    )


from optax._src import linear_algebra


def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))
    # logged_data["grad"] = driver._loss_grad
    return True


n_particles = 10
n_dims = 1 * n_particles

# density = 2.19 * 10 ** 28
density1 = 0.3 * 10 ** 10
rm = 2.9673 * 10 ** (-10)

L = n_particles / (density1 * rm)
print(L)

hib = nk.hilbert.ContinuousBoson(N=n_particles, L=(L,), pbc=True)

sab = nk.sampler.MetropolisGaussian(hib, sigma=0.05, n_chains=16, n_sweeps=10)

model = nk.models.DS(
    k=10,
    L=L,
    sdim=1,
    layers_phi=3,
    layers_rho=3,
    features_phi=(32, 32, 1),
    features_rho=(32, 32, 1),
    dtype=jnp.float64,
)

ha = nk.operator.KineticPotential(hib, potential)

vs = nk.vqs.MCState(sab, model, n_samples=50000, n_discard=2000)
lr = 0.01

from optax._src import clipping
from optax import sgd
from optax import chain

# op = chain(clipping.clip_by_global_norm(1.),
#           sgd(lr))
op = nk.optimizer.Sgd(lr)
# from functools import partial
# import jax
# sr = nk.optimizer.SR(diag_shift=0.01, solver=partial(jax.scipy.sparse.linalg.cg, maxiter=10000))
sr = nk.optimizer.SR(diag_shift=0.005)

# import flax
# with open("/home/gabriel/Documents/PhD/netket/Examples/Continuous/DeepSets/HE10_Jastrow.mpack", 'rb') as file:
#    vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
# print(vs.variables)


# print(vs.expect_and_grad(ha))
# print(vs.variables)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
# Run the optimization for 300 iterationst

gs.run(n_iter=250, callback=mycb, out="HE10_blabla")
# print(vs.variables)
