from typing import Any
import math

from numba import jit

import numpy as np
from flax import struct

from netket.legacy import random as _random
from netket.operator import AbstractOperator


from ..metropolis import MetropolisRule


class HamiltonianRuleState:
    n_conn: int
    """Preallocated array for sampler state"""


@struct.dataclass
class HamiltonianRuleNumpy(MetropolisRule):
    Ô: Any = struct.field(pytree_node=False)

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.Ô, AbstractOperator):
            raise TypeError(
                "Argument to HamiltonianRuleNumpy must be a valid operator.".format(
                    type(operator)
                )
            )

    def init_state(rule, sampler, machine, params, key):
        return None

    def reset(rule, sampler, machine, parameters, sampler_state):
        return None

    def transition(rule, sampler, machine, parameters, state, rng, σ):
        σ = state.σ
        σ1 = state.σ1
        log_prob_corr = state.log_prob_corr

        sections = np.empty(σ.shape[0], dtype=np.int32)
        σp = rule.Ô.get_conn_flattened(σ, sections)[0]

        rand_vec = rng.uniform(0, 1, size=σ.shape[0])

        _choose(σp, sections, σ1, log_prob_corr, rand_vec)
        rule.Ô.n_conn(σ1, sections)
        log_prob_corr -= np.log(sections)


@jit(nopython=True)
def _choose(states, sections, out, w, rand_vec):
    low_range = 0
    for i, s in enumerate(sections):
        n_rand = int(rand_vec[i] * s)
        out[i] = states[n_rand]
        w[i] = math.log(s - low_range)
        low_range = s
