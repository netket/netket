# Copyright 2024 The NetKet Authors - All rights reserved.
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


import flax.linen as nn
import jax.numpy as jnp
import numpy as np

import netket as nk

from .. import common


class M1(nn.Module):
    n_h: int

    @nn.compact
    def __call__(self, x):
        n_v = x.shape[-1]
        W = self.param(
            "weights", nn.initializers.normal(), (self.n_h, n_v), jnp.complex128
        )

        y = jnp.einsum("ij,...j", W, x)
        return jnp.sum(nk.nn.activation.reim_selu(y), axis=-1)


class M2(nn.Module):
    n_h: int

    @nn.compact
    def __call__(self, x):
        n_v = x.shape[-1]
        Wr = self.param(
            "weights_re", nn.initializers.normal(), (self.n_h, n_v), jnp.float64
        )
        Wi = self.param(
            "weights_im", nn.initializers.normal(), (self.n_h, n_v), jnp.float64
        )

        W = Wr + 1j * Wi

        y = jnp.einsum("ij,...j", W, x)
        return jnp.sum(nk.nn.activation.reim_selu(y), axis=-1)


class M3(nn.Module):
    n_h: int

    @nn.compact
    def __call__(self, x):
        n_v = x.shape[-1]
        W0r = self.param(
            "weights_re",
            nn.initializers.normal(),
            (self.n_h // 2, n_v),
            jnp.float64,
        )
        W0i = self.param(
            "weights_im",
            nn.initializers.normal(),
            (self.n_h // 2, n_v),
            jnp.float64,
        )
        W1 = self.param(
            "weights",
            nn.initializers.normal(),
            (self.n_h // 2, n_v),
            jnp.complex128,
        )
        W = jnp.concatenate([W0r + 1j * W0i, W1], axis=0)
        y = jnp.einsum("ij,...j", W, x)
        return jnp.sum(nk.nn.activation.reim_selu(y), axis=-1)


def test_forces_gradient_rule():
    nh = 8
    ma1 = M1(nh)
    ma2 = M2(nh)
    ma3 = M3(nh)
    hi = nk.hilbert.Spin(1 / 2, N=4)
    ha = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=0.5)
    samp = nk.sampler.ExactSampler(hi)
    vs1 = nk.vqs.MCState(samp, model=ma1, n_samples=1024, sampler_seed=1234, seed=1234)
    vs2 = nk.vqs.MCState(samp, model=ma2, n_samples=1024, sampler_seed=1234, seed=1234)
    vs3 = nk.vqs.MCState(samp, model=ma3, n_samples=1024, sampler_seed=1234, seed=1234)

    vs1.parameters = {
        "weights": vs2.parameters["weights_re"] + 1j * vs2.parameters["weights_im"]
    }
    vs3.parameters = {
        "weights": vs2.parameters["weights_re"][nh // 2 :]
        + 1j * vs2.parameters["weights_im"][nh // 2 :],
        "weights_re": vs2.parameters["weights_re"][: nh // 2],
        "weights_im": vs2.parameters["weights_im"][: nh // 2],
    }

    np.testing.assert_allclose(vs1.to_array(), vs2.to_array())
    np.testing.assert_allclose(vs1.to_array(), vs3.to_array())

    _, f1 = vs1.expect_and_forces(ha)
    _, f2 = vs2.expect_and_forces(ha)
    _, f3 = vs3.expect_and_forces(ha)

    _, g1 = vs1.expect_and_grad(ha)
    _, g2 = vs2.expect_and_grad(ha)
    _, g3 = vs3.expect_and_grad(ha)

    np.testing.assert_allclose(g1["weights"], 2 * f1["weights"])

    np.testing.assert_allclose(g2["weights_re"], 2 * np.real(f2["weights_re"]))
    np.testing.assert_allclose(g2["weights_im"], 2 * np.real(f2["weights_im"]))

    np.testing.assert_allclose(g3["weights"], 2 * f3["weights"])
    np.testing.assert_allclose(g3["weights_re"], 2 * np.real(f3["weights_re"]))
    np.testing.assert_allclose(g3["weights_im"], 2 * np.real(f3["weights_im"]))

    # check that the gradient of the real-param and complex-param model are the same
    np.testing.assert_allclose(g1["weights"], g2["weights_re"] + 1j * g2["weights_im"])
    # check that the gradient of the mixed-param and complex-param model are the same
    np.testing.assert_allclose(
        g1["weights"],
        jnp.concatenate(
            [g3["weights_re"] + 1j * g3["weights_im"], g3["weights"]], axis=0
        ),
    )


@common.skipif_sharding  # no jax version of LocalLiouvillian
def test_forces_gradient_rule_ldagl():
    nh = 8
    ma1 = M1(nh)
    ma2 = M2(nh)
    ma3 = M3(nh)
    hi = nk.hilbert.Spin(1 / 2, N=3)
    hi2 = nk.hilbert.DoubledHilbert(hi)
    ha = nk.operator.Ising(hi, nk.graph.Chain(3), h=0.5)
    jump_ops = [nk.operator.spin.sigmam(hi, i) for i in range(3)]
    lind = nk.operator.LocalLiouvillian(ha.to_local_operator(), jump_ops)
    ha = nk.operator.Squared(lind)

    samp = nk.sampler.ExactSampler(hi2)
    vs1 = nk.vqs.MCMixedState(
        samp, model=ma1, n_samples=1024, sampler_seed=1234, seed=1234
    )
    vs2 = nk.vqs.MCMixedState(
        samp, model=ma2, n_samples=1024, sampler_seed=1234, seed=1234
    )
    vs3 = nk.vqs.MCMixedState(
        samp, model=ma3, n_samples=1024, sampler_seed=1234, seed=1234
    )

    vs1.parameters = {
        "weights": vs2.parameters["weights_re"] + 1j * vs2.parameters["weights_im"]
    }
    vs3.parameters = {
        "weights": vs2.parameters["weights_re"][nh // 2 :]
        + 1j * vs2.parameters["weights_im"][nh // 2 :],
        "weights_re": vs2.parameters["weights_re"][: nh // 2],
        "weights_im": vs2.parameters["weights_im"][: nh // 2],
    }

    np.testing.assert_allclose(vs1.to_array(), vs2.to_array())
    np.testing.assert_allclose(vs1.to_array(), vs3.to_array())

    _, g1 = vs1.expect_and_grad(ha)
    _, g2 = vs2.expect_and_grad(ha)
    _, g3 = vs3.expect_and_grad(ha)

    # check that the gradient of the real-param and complex-param model are the same
    np.testing.assert_allclose(g1["weights"], g2["weights_re"] + 1j * g2["weights_im"])
    # check that the gradient of the mixed-param and complex-param model are the same
    np.testing.assert_allclose(
        g1["weights"],
        jnp.concatenate(
            [g3["weights_re"] + 1j * g3["weights_im"], g3["weights"]], axis=0
        ),
    )
