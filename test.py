from typing import Callable
import jax
import jax.numpy as jnp
from netket.utils.numbers import dtype as _dtype

from netket.utils.types import Array
from netket.experimental.dynamics import IntegratorState, AbstractSolver
from netket.utils import struct


class MyState(struct.Pytree):
    info : str = 'empty'

    def __init__(
        self,
        integrator_state
    ):
        self.info = str(integrator_state.step_no_total)

    def __repr__(self):

        return  f'MYSTATE({self.info})' 

class EulerIntegrator(AbstractSolver):
    r"""
    Class representing the Taleau of the Euler method which, given the ODE :math:`dy/dt = F(t, y)`, updates the solution as

    .. math::
        y_{t+dt} = y_t + dt f(t, y_t)

    """
    def __init__(self, dt):
        return super().__init__(dt=dt, adaptive=False)
    
    def step(
        self, f: Callable, dt: float, t: float, y_t, state
    ):
        """Perform one fixed-size Euler step from `t` to `t + dt`."""
        dy = f(t, y_t)

        y_tp1 = jax.tree_util.tree_map(
            lambda y_t, dy: y_t
            + jnp.asarray(dt, dtype=y_t.dtype) * jnp.asarray(dy, dtype=y_t.dtype),
            y_t,
            dy,
        )

        return y_tp1, state.replace(info=str(t))
    
    def _init_state(self, integrator_state):
        return MyState(integrator_state=integrator_state)
    


import netket as nk
import netket.experimental as nkx

for solver in [EulerIntegrator(dt=1e-2), nkx.dynamics.RK4(dt=1e-2), nkx.dynamics.RK45(dt=1e-2, adaptive=True, rtol=1e-3, atol=1e-1)]:


    N = 2
    hi = nk.hilbert.Spin(1 / 2, N)
    vs = nk.vqs.FullSumState(hi, model=nk.models.LogStateVector(hi))
    H = -sum([nk.operator.spin.sigmax(hi, i) for i in range(N)]) + sum([nk.operator.spin.sigmaz(hi, i) for i in range(N)])
    te = nkx.TDVP(
        operator=H,
        variational_state=vs,
        integrator=solver,
        t0=0.0,
        qgt=nk.optimizer.qgt.QGTJacobianDense(diag_shift=0, holomorphic=True),
        error_norm="qgt",
        linear_solver=nk.optimizer.solver.svd(rcond=1e-8),
    )
    print(te.integrator)
    # print(te.integrator._state)

    # def cb(step, log_data, driver):
    #     print(driver.integrator._state)
    #     return True
    te.run(1.0, )

    print(te.integrator)
    # print(te.solver)