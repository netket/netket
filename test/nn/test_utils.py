# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from itertools import combinations_with_replacement
from functools import reduce
import operator

import jax.numpy as jnp
import numpy as np
import jax
from jax.nn.initializers import normal

import netket as nk
from netket.nn import binary_encoding

from .. import common  # noqa: F401

SEED = 111


@pytest.fixture(
    params=[pytest.param(M, id=f"Fock(M={M})") for M in [0, 2, 3, 4, 5, 6, 8]]
)
def vstate(request):
    M = request.param
    # keep this a prime number so we get different sizes on every rank...
    hi = nk.hilbert.Fock(M, 1)

    ma = nk.models.RBM(
        alpha=1,
        param_dtype=float,
        hidden_bias_init=normal(),
        visible_bias_init=normal(),
    )

    return nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("chunk_size", [None, 4])
def test_to_array(vstate, normalize, chunk_size):
    vstate.chunk_size = chunk_size

    psi = vstate.to_array(normalize=normalize)

    if normalize:
        np.testing.assert_allclose(jnp.linalg.norm(psi), 1.0)

    psi_norm = psi / jnp.linalg.norm(psi)

    assert psi.shape == (vstate.hilbert.n_states,)

    x = vstate.hilbert.all_states()
    psi_exact = jnp.exp(vstate.log_value(x))
    psi_exact = psi_exact / jnp.linalg.norm(psi_exact)

    np.testing.assert_allclose(psi_norm, psi_exact)


@pytest.fixture(
    params=[pytest.param(M, id=f"Fock(M={M})") for M in [0, 2, 3, 4, 5, 6, 8]]
)
def vstate_rho(request):
    M = request.param
    # keep this a prime number so we get different sizes on every rank...
    hi = nk.hilbert.Fock(M, 1)

    ma = nk.models.NDM()

    return nk.vqs.MCMixedState(
        nk.sampler.MetropolisLocal(nk.hilbert.DoubledHilbert(hi)),
        ma,
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_to_matrix(vstate_rho, normalize):
    rho = vstate_rho.to_matrix(normalize=normalize)

    if normalize:
        np.testing.assert_allclose(jnp.trace(rho), 1.0)

    rho_norm = rho / jnp.trace(rho)

    assert rho.shape == (
        vstate_rho.hilbert.physical.n_states,
        vstate_rho.hilbert.physical.n_states,
    )

    x = vstate_rho.hilbert.all_states()
    rho_exact = jnp.exp(vstate_rho.log_value(x)).reshape(rho.shape)
    rho_exact = rho_exact / jnp.trace(rho_exact)

    np.testing.assert_allclose(rho_norm, rho_exact)


def _get_tls_hilbert_space(_type: str) -> nk.hilbert.DiscreteHilbert:
    if _type == "spin":
        return nk.hilbert.Spin(s=1 / 2, N=1)
    elif _type == "qubit":
        return nk.hilbert.Qubit(N=1)
    else:
        raise ValueError("Supported types are 'spin' and 'qubit'")


def _create_hilbert_space(shape) -> nk.hilbert.DiscreteHilbert:
    n_tls = shape.count(2)
    tls_hilbert = ["spin", "qubit"]
    for tlss in combinations_with_replacement(tls_hilbert, n_tls):
        hilberts = []
        m = 0
        for n in shape:
            if n == 2:
                hilberts.append(_get_tls_hilbert_space(tlss[m]))
                m += 1
            else:
                hilberts.append(nk.hilbert.Fock(n_max=n - 1, N=1))
        yield reduce(operator.mul, hilberts[1:], hilberts[0])


def _int_to_binary_list(x, n_total_bits):
    b = format(int(max(0, x)), "b")
    nb = len(b)
    zeros = n_total_bits - nb
    return [0] * zeros + [int(br) for br in b]


def _state_to_binary_list(random_state, bits_per_site):
    return [
        _int_to_binary_list(x, nbits) for (x, nbits) in zip(random_state, bits_per_site)
    ]


@common.skipif_mpi
@pytest.mark.parametrize("hilbert_shape", [(2,), (2, 2), (2, 3), (4, 3, 2)])
def test_binary_encoding(hilbert_shape):
    for hilbert in _create_hilbert_space(hilbert_shape):
        shape = tuple(hilbert.shape)
        bits_per_site = [int(np.ceil(np.log2(s))) for s in shape]
        total_bits = sum(bits_per_site)
        random_state = hilbert.random_state(key=jax.random.PRNGKey(0))
        encoded_with_hilbert = binary_encoding(hilbert, random_state)
        assert total_bits == encoded_with_hilbert.size
        random_state_i = hilbert.states_to_local_indices(random_state)
        desired_state = sum(_state_to_binary_list(random_state_i, bits_per_site), [])
        np.testing.assert_allclose(encoded_with_hilbert, desired_state)
