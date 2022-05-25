# Copyright 2022 The NetKet Authors - All rights reserved.
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

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def logsumexp_cplx(a, b=None, **kwargs):
    """Compute the log of the sum of exponentials of input elements, always returning a
    complex number.

    Equivalent to, but more numerically stable than, `np.log(np.sum(b*np.exp(a)))`.
    If the optional argument `b` is omitted, `np.log(np.sum(np.exp(a)))` is returned.

    Wraps `jax.scipy.special.logsumexp` but uses `return_sign=True` if both `a` and `b`
    are real numbers in order to support `b<0` instead of returning `nan`.

    See the JAX function for details of the calling sequence;
    `return_sign` is not supported.
    """
    if jnp.iscomplexobj(a) or jnp.iscomplexobj(b):
        # logsumexp uses complex algebra anyway
        return logsumexp(a, b=b, **kwargs)
    else:
        a, sgn = logsumexp(a, b=b, **kwargs, return_sign=True)
        a = a + jnp.where(sgn < 0, 1j * jnp.pi, 0j)
        return a
