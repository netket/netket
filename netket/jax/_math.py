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


import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from netket.utils.types import Array

from ._utils_dtype import dtype_complex


def logsumexp_cplx(a: Array, b: Array | None = None, **kwargs) -> jax.Array:
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


def logdet_cmplx(A: Array) -> jax.Array:
    r"""Log-determinant, with automatic upconversion to a complex
    output dtype in order to encode the sign.

    This is a thin wrapper on top of {func}`jax.numpy.linalg.slogdet`.
    The mathematical formula is:

    .. math::

        \log(|A |)

    Args:
        A: A square matrix, or batch of matrices. The shape should be
            `(..., N, N)`

    Return:
        A scalar or batch of scalars with the smallest complex dtype
        computed from the input dtype. If the input has shape
        `(..., N, N)` the output has shape `(...)`

    """
    sign, logabsdet = jnp.linalg.slogdet(A)
    cplx_type = dtype_complex(A.dtype)
    return logabsdet.astype(cplx_type) + jnp.log(sign.astype(cplx_type))
