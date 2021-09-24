import netket as nk
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

"""
import json

file20 = open('CS_50particles_vanilla_b20_final.log')
file20_train = open('CS_50particles_vanilla_b20.log')
file11 = open('CS_50particles_vanilla_b11_final.log')
file15 = open('CS_50particles_vanilla_b15_final.log')
file30 = open('CS_50particles_vanilla_b30_final.log')
data20 = jnp.array(json.load(file20)['Energy']['Mean'])
data20_train = jnp.array(json.load(file20_train)['Energy']['Mean'])
data11 = jnp.array(json.load(file11)['Energy']['Mean'])
data15 = jnp.array(json.load(file15)['Energy']['Mean'])
data30 = jnp.array(json.load(file30)['Energy']['Mean'])

energy = jnp.array([jnp.mean(data11), jnp.mean(data15), jnp.mean(data20), jnp.mean(data30)])
eenergy = jnp.array([jnp.std(data11), jnp.std(data15), jnp.std(data20), jnp.std(data30)])
refenergy = jnp.array([2486.967590997499, 4624.526512185, 8221.3804661074, 18498.10604874173])
exact_energy = lambda x: 1/6 * (jnp.pi/10.)**2 * x**2 *(50**2-1)
beta = jnp.array([1.1,1.5,2.0,3.0])
#energy = jnp.concatenate((jnp.array(data1['Energy']['Mean']),jnp.array(data2['Energy']['Mean']),jnp.array(data3['Energy']['Mean']))).reshape(-1)
print(energy)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset

fig, axes = plt.subplots(1, figsize=(10,10))
axes.plot(jnp.arange(0,data20_train.size), data20_train/50., color='b', label='VMC')
axes.axhline(y=refenergy[2]/50., color='r', label='exact')
axes.grid()
axes.legend()
axes.set_xlabel('Iteration')
axes.set_ylabel('Energy <E> / N')

axes1 = plt.axes([0,0,1,1])
ip = InsetPosition(axes, [0.37, 0.35, 0.6, 0.5])
axes1.set_axes_locator(ip)
axes1.ticklabel_format(useOffset=False)
axes1.plot(jnp.arange(10,data20_train.size), data20_train[10:]/50., color='b', label='VMC')
axes1.axhline(y=refenergy[2]/50., color='r', label='exact')
axes1.grid()

plt.savefig('CS50_energy_b20.png')
interval = jnp.arange(0,3.1,0.01)
fig, axes = plt.subplots(1, figsize=(10,10))
axes.plot(interval, exact_energy(interval) , color='r', label='exact')
axes.errorbar(beta, energy/50., yerr=eenergy/50., fmt='o', color='b', label='VMC')
axes.grid()
axes.legend()
axes.set_xlabel('b')
axes.set_ylabel('Energy <E> / N')

plt.savefig('CS50_energy_vs_beta.png')

"""


def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    # logged_data["grad"] = driver._loss_grad
    return True


def minimum_distance(x, sdim):

    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    # pdist = jnp.remainder(dist + L / 2., L) - L / 2.
    # basis = jnp.array(jnp.meshgrid([0, L, -L], [0, L, -L], [0, L, -L])).T.reshape(-1, 3)
    # distances = (pdist[:, jnp.newaxis, :] + basis).reshape(-1, sdim)
    return jnp.linalg.norm(distances, axis=1)


def potential(x):
    dis = minimum_distance(x, 1)

    dis = jnp.sin(jnp.pi / L * dis) ** 2
    beta = 2.0
    g = (jnp.pi / L) ** 2 * beta * (beta - 1)

    return jnp.sum(g / dis)


n_particles = 5
L = 10.0

hib = nk.hilbert.ContinuousBoson(N=n_particles, L=(L,), pbc=True)

sab = nk.sampler.MetropolisGaussian(hib, sigma=0.1, n_chains=16, n_sweeps=10)

model = nk.models.DS(
    k=2,
    L=L,
    sdim=1,
    layers_phi=1,
    layers_rho=2,
    features_phi=(32,),
    features_rho=(32, 1),
    dtype=jnp.float64,
)

ha = nk.operator.KineticPotential(hib, potential)

vs = nk.vqs.MCState(sab, model, n_samples=100000, n_discard=2000)
"""
import flax
with open("/home/gabriel/Documents/PhD/netket/Examples/Continuous/DeepSets/CS_50particles_vanilla_b20.mpack", 'rb') as file:
    vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
"""
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
# Run the optimization for 300 iterations
# print(vs.expect_and_grad(ha))
# sprint(vs.variables)
gs.run(n_iter=1000, callback=mycb, out="test")
# gs.state.n_samples = 50000
# gs.run(n_iter=10, callback=mycb, out="CS_50particles_vanilla")
