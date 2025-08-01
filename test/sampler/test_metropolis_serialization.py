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

from jax import numpy as jnp
from flax import serialization
from jax.experimental import multihost_utils

import pytest
import numpy as np
import jax

import netket as nk


nk.config.update("NETKET_EXPERIMENTAL", True)
np.random.seed(1234)

WEIGHT_SEED = 1234
SAMPLER_SEED = 15324


def _allgather(x):
    if nk.config.netket_experimental_sharding and jax.process_count() > 1:
        x = multihost_utils.process_allgather(x)
    return x


@pytest.mark.parametrize(
    "sampler", [nk.sampler.MetropolisLocal, nk.sampler.ParallelTemperingLocal]
)
@pytest.mark.parametrize("key_type", ["PRNGKey", "key"])
def test_metropolis_serialization(key_type, sampler, tmp_path_distributed):
    rank = jax.process_index()
    n_nodes = jax.process_count()
    keyT = jax.random.PRNGKey if key_type == "PRNGKey" else jax.random.key

    tmp_file_name = tmp_path_distributed / "data.mpack"
    print(f"r{rank}/{n_nodes} - saving to: {tmp_file_name}")

    hi = nk.hilbert.Spin(0.5, 10)
    ma = nk.models.RBM()
    pars = ma.init(jax.random.key(WEIGHT_SEED), jnp.zeros((5, 10)))

    # 120 = (2**3)* 3 * 5 which are all prime factors reasonable for
    # a CI test. Doubt anybody using more than 6 nodes in any case...
    sa = sampler(hi, n_chains=12)
    sampler_state = sa.init_state(ma, pars, keyT(SAMPLER_SEED))
    sampler_state = sa.reset(ma, pars, state=sampler_state)

    state_dict = serialization.to_state_dict(sampler_state)

    # if we must do it, we must replicate
    # if nk.config.netket_experimental_sharding and nk.config.netket_experimental_sharding_fast_serialization:
    #     def _fun(x):
    #         if isinstance(x, jax.Array) and not x.is_fully_replicated:
    #             return jax.lax.with_sharding_constraint(x, x.sharding.replicate())
    #         return x
    #     state = jax.tree.map(_fun, state)

    if jax.process_index() == 0:
        bindata = serialization.msgpack_serialize(state_dict)
        with open(tmp_file_name, "wb") as f:
            f.write(bindata)

    if nk.config.netket_experimental_sharding:
        multihost_utils.sync_global_devices("ciao")

    # load data
    with open(tmp_file_name, "rb") as f:
        bindata = f.read()

    # Create another sampler state
    for n_readout_chains in [12, 8, 6]:
        print("n_readout_chains: ", n_readout_chains)
        sa = sampler(hi, n_chains=n_readout_chains)
        sampler_state_2 = sa.init_state(ma, pars, SAMPLER_SEED + 100)

        state_dict_loaded = serialization.msgpack_restore(bindata)
        sampler_state_2 = serialization.from_state_dict(
            sampler_state_2, state_dict_loaded
        )

        if nk.config.netket_experimental_sharding:
            if isinstance(
                sampler_state_2.σ.sharding, jax.sharding.SingleDeviceSharding
            ):
                pass
            else:
                assert sampler_state_2.σ.sharding.is_equivalent_to(
                    sampler_state.σ.sharding, sampler_state.σ.ndim
                )
            assert bool(
                jnp.all(sampler_state_2.σ == sampler_state.σ[: sa.n_batches, :])
            )
        else:
            assert sampler_state_2.σ.sharding.shape == (len(jax.devices()), 1)
        assert bool(jnp.all(sampler_state_2.σ == sampler_state.σ[: sa.n_batches, :]))

        # verify that it works
        sa.samples(ma, pars, state=sampler_state_2)
