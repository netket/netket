import netket as nk
import jax
from jax.flatten_util import ravel_pytree
import numpy as np
import netket.nn as nknn

hi =nk.hilbert.Spin(0.5, 4)
ma = nk.models.RBM(alpha=1, dtype=float,kernel_init=nknn.initializers.normal(0.5))
sa = nk.sampler.ExactSampler(hi)

n_samples=1600000
vs = nk.vqs.MCState(sa, ma,n_samples=n_samples)

qgt = vs.quantum_geometric_tensor()


all_states=jax.numpy.asarray(hi.all_states())

def NormPsi(pars):
    return jax.numpy.sqrt(jax.numpy.sum(jax.numpy.abs(jax.numpy.exp(ma.apply(pars,all_states)))**2))


def Psi(pars,x):
    return jax.numpy.exp(ma.apply(pars,x))/NormPsi(pars)

def GradPsiSingle(pars,x):
    return ravel_pytree(jax.grad(Psi,argnums=0)(pars,x))[0]


def GradPsi(pars,x):
    return jax.vmap(GradPsiSingle,(None,0))(pars,x)



PsiVec=Psi(vs.variables,all_states)
GVec=GradPsi(vs.variables,all_states)

#The exact qgt is <GradPsi|GradPsi> -<GradPsi|Psi><Psi|GradPsi>
qgt_ex=(GVec.conjugate().transpose()@GVec) -(GVec.conjugate().transpose()@PsiVec)*(PsiVec.conjugate().transpose()@GVec)

qgt_est=qgt.to_dense()

np.testing.assert_allclose(qgt_est,qgt_ex,atol=10.0/np.sqrt(n_samples))
