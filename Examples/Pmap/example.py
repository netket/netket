import os
os.environ["NETKET_EXPERIMENTAL_PMAP"] = "4"

import netket as nk
import jax
print(jax.devices())

import jax.numpy as jnp
from functools import partial

hi = nk.hilbert.Spin(0.5, 3)
ma = nk.models.RBM()
sa = nk.sampler.MetropolisLocal(hi)
vs = nk.vqs.MCState(sa, ma)

ha = nk.operator.Ising(hi, nk.graph.Chain(hi.size), h=1.0)

vs.samples
r=vs.expect(ha)
print(r)