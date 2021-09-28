
from plum import dispatch

import jax.numpy as jnp

def _ode_addsteps(integrator, fun=None,
                  always_calc_begin = False,
                  allow_calc_end = True,
                  force_calc_end = False):
    if fun is None:
        fun = integrator.f

    return addsteps(integrator.k, integrator.tprev, integrator.uprev, integrator.u, integrator.dt, 
                    integrator.cache, integrator.p, always_calc_begin, allow_calc_end, force_calc_end)    

def addsteps(k, t, uprev, u, dt, f, p, cache, always_calc_begin, allow_calc_end, force_calc_end):
    if length(k) < 2:
        raise ValueError("size is wrong")

    return k

@dispatch
def ode_interpolant(theta, integrator, idxs, deriv):
    return ode_interpolant(theta, integrator.dt, integrator.uprev, integrator.u, integrator.k, integrator.cache, idxs, deriv)

@dispatch
def ode_interpolant(theta, dt, y0, y1, k, cache, idxs, deriv):
    return hermite_interpolant(theta, dt, y0, y1, k, cache, idxs, deriv)

def hermite_interpolant(Θ,dt,y0,y1,k,cache,idxs:None, deriv): # Default interpolant is Hermite
    """
    Hairer Norsett Wanner Solving Ordinary Differential Euations I - Nonstiff Problems Page 190

    Herimte Interpolation, chosen if no other dispatch for ode_interpolant
    """
    return (1-Θ)*y0+Θ*y1+Θ*(Θ-1)*((1-2*Θ)*(y1-y0)+(Θ-1)*dt*k.at[1].get() + Θ*dt*k.at[2].get())
