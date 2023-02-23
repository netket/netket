import netket as nk

import jax
import jax.numpy as jnp

ma = nk.models.RBM(alpha=2)

hi = nk.hilbert.POVMTethra(N=2)

sa = nk.sampler.MetropolisLocal(hi)

vs = nk.vqs.MCPOVMState(sa, ma)