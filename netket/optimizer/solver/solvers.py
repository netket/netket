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


import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from netket.utils.api_utils import partial_from_kwargs
from netket.utils.deprecation import warn_deprecation
from netket.utils.optional_deps import import_optional_dependency
from netket.utils.citations import reference


@partial_from_kwargs
def pinv_smooth(
    A,
    b,
    *,
    rtol: float = 1e-14,
    rtol_smooth: float = 1e-14,
    x0=None,
    rcond: float = None,
    rcond_smooth: float = None,
    return_eigvals: bool = False,
):
    r"""
    Solve the linear system by building a pseudo-inverse from the
    eigendecomposition obtained from :func:`jax.numpy.linalg.eigh`.

    The eigenvalues :math:`\lambda_i` smaller than
    :math:`r_\textrm{cond} \lambda_\textrm{max}` are truncated (where
    :math:`\lambda_\textrm{max}` is the largest eigenvalue).

    The eigenvalues are further smoothed with another filter, originally introduced in
    `Medvidovic, Sels arXiv:2212.11289 (2022) <https://arxiv.org/abs/2212.11289>`_,
    given by the following equation

    .. math::

        \tilde\lambda_i^{-1}=\frac{\lambda_i^{-1}}{1+\big(\epsilon\frac{\lambda_\textrm{max}}{\lambda_i}\big)^6}


    .. note::

        In general, we found that this custom implementation of
        the pseudo-inverse outperform
        jax's :func:`~jax.numpy.linalg.pinv`. This might be
        because :func:`~jax.numpy.linalg.pinv` internally calls
        :obj:`~jax.numpy.linalg.svd`, while this solver internally
        uses :obj:`~jax.numpy.linalg.eigh`.

        For that reason, we suggest you use this solver instead of
        :obj:`~netket.optimizer.solver.pinv`.


    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.


    Args:
        A: LinearOperator (matrix)
        b: vector or Pytree
        rtol : Relative tolerance for small singular values of :code:`A`. For
            the purposes of rank determination, singular values are treated
            as zero if they are smaller than `rtol` times the largest
            singular value of :code:`A`.
        rtol_smooth : Regularization parameter used with a similar effect to `rtol`
            but with a softer curve. See :math:`\epsilon` in the formula
            above.
        rcond: (deprecated) Alias for `rtol`. Will be removed in a future release.
        rcond_smooth: (deprecated) Alias for `rtol_smooth`. Will be removed in a future release.
        return_eigvals: If True, also return the eigenvalues of A.
    """
    del x0

    if rcond is not None:
        warn_deprecation(
            "The 'rcond' argument is deprecated and will be removed in a future release. "
            "Please use 'rtol' instead."
        )
        rtol = rcond

    if rcond_smooth is not None:
        warn_deprecation(
            "The 'rcond_smooth' argument is deprecated and will be removed in a future release. "
            "Please use 'rtol_smooth' instead."
        )
        rtol_smooth = rcond

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    Σ, U = jnp.linalg.eigh(A)

    # Discard eigenvalues below numerical precision
    Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rtol, jnp.reciprocal(Σ), 0.0)

    # Set regularizer for singular value cutoff
    regularizer = 1.0 / (1.0 + (rtol_smooth / jnp.abs(Σ / Σ[-1])) ** 6)

    Σ_inv = Σ_inv * regularizer

    x = U @ (Σ_inv * (U.conj().T @ b))

    rank = jnp.sum(Σ > rtol * Σ[-1])
    cond_number = Σ[-1] / Σ[0]
    info = {
        "eval_min": Σ[0],
        "eval_max": Σ[-1],
        "rank": rank,
        "cond_number": cond_number,
    }
    if return_eigvals:
        info["evals"] = Σ

    return unravel(x), info


@partial_from_kwargs
def pinv(
    A,
    b,
    *,
    rtol: float = 1e-12,
    x0=None,
    rcond: float = None,
    return_eigvals: bool = False,
):
    """
    Solve the linear system using jax's implementation of the
    pseudo-inverse.

    Internally it calls :func:`~jax.numpy.linalg.pinv` which
    uses a :func:`~jax.numpy.linalg.svd` decomposition with
    the same value of **rtol**.

    .. note::

        In general, we found that our custom implementation of
        the pseudo-inverse
        :func:`netket.optimizer.solver.pinv_smooth` (which
        internally uses hermitian diagonaliation) outperform
        jax's :func:`~jax.numpy.linalg.pinv`.

        For that reason, we suggest to use
        :func:`~netket.optimizer.solver.pinv_smooth` instead of
        :obj:`~netket.optimizer.solver.pinv`.


    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    The diagonal shift on the matrix can be 0 and the
    **rtol** variable can be used to truncate small
    eigenvalues.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rtol: Cut-off ratio for small singular values of :code:`A`.
        rcond: (deprecated) Alias for `rtol`. Will be removed in a future release.
    """
    del x0

    if rcond is not None:
        warn_deprecation(
            "The 'rcond' argument is deprecated and will be removed in a future release. "
            "Please use 'rtol' instead."
        )
        rtol = rcond

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    A_inv = jnp.linalg.pinv(A, rtol=rtol, hermitian=True)

    x = jnp.dot(A_inv, b)

    return unravel(x), None


@partial_from_kwargs
def svd(A, b, *, rcond=None, x0=None):
    """
    Solve the linear system using Singular Value Decomposition.
    The diagonal shift on the matrix should be 0.

    Internally uses :func:`jax.numpy.linalg.lstsq`.

    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rcond: The condition number
    """
    del x0

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    x, residuals, rank, s = jnp.linalg.lstsq(A, b, rcond=rcond)

    info = {
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
    }

    return unravel(x), info


@partial_from_kwargs
def cholesky(A, b, *, lower=False, x0=None):
    """
    Solve the linear system using a Cholesky Factorisation.
    The diagonal shift on the matrix should be 0.

    Internally uses :func:`jax.numpy.linalg.cho_solve`.

    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        lower: if True uses the lower half of the A matrix
        x0: unused
    """

    del x0

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    c, low = jsp.linalg.cho_factor(A, lower=lower)
    x = jsp.linalg.cho_solve((c, low), b)
    return unravel(x), None


@partial_from_kwargs
def LU(A, b, *, trans=0, x0=None):
    """
    Solve the linear system using a LU Factorisation.
    The diagonal shift on the matrix should be 0.

    Internally uses :func:`jax.numpy.linalg.lu_solve`.

    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        lower: if True uses the lower half of the A matrix
        x0: unused
    """

    del x0

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    lu, piv = jsp.linalg.lu_factor(A)
    x = jsp.linalg.lu_solve((lu, piv), b, trans=0)
    return unravel(x), None


# I believe this internally uses a smarter/more efficient way to
# do cholesky
@partial_from_kwargs
def solve(A, b, *, assume_a="pos", x0=None):
    """
    Solve the linear system.
    The diagonal shift on the matrix should be 0.

    Internally uses :func:`jax.numpy.solve`.

    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        lower: if True uses the lower half of the A matrix
        x0: unused
    """
    del x0

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    x = jsp.linalg.solve(A, b, assume_a="pos")
    return unravel(x), None


@reference(
    "Wiersema2026jaxmg",
    condition="If using cholesky_distributed solver",
    message="This work used the JAXMg distributed linear solver described in Ref.",
)
@partial_from_kwargs
def cholesky_distributed(A, b, *, local_tile_size=None, x0=None):
    """
    Solve the linear system using a distributed Cholesky factorization.

    This is the distributed/multi-GPU equivalent of :func:`~netket.optimizer.solver.cholesky`.
    It uses jaxmg's `potrs` function (LAPACK POTRF+POTRS) which performs a
    Cholesky factorization followed by triangular solves, optimized for
    sharded arrays with automatic tiling and communication optimization.

    .. note::

        This solver requires the optional `jaxmg` package to be installed:

        .. code-block:: bash

            pip install jaxmg

        If jaxmg is not installed, an ImportError will be raised.

    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    .. note::

        This solver is specifically designed for use with
        `config.netket_experimental_sharding = True` and expects the input
        matrix A to be sharded with `P("S", None)`.

        For single-device or small-scale computations, use
        :func:`~netket.optimizer.solver.cholesky` instead, which has less
        overhead. Use `cholesky_distributed` when:

        - Running on multiple GPUs with sharded arrays
        - NTK/QGT matrix is very large and doesn't fit on a single device
        - You need to minimize communication in distributed settings

    Args:
        A: the matrix A in Ax=b (should be positive definite, sharded)
        b: the vector b in Ax=b
        local_tile_size: Tile size for matrix A. Controls the tiling strategy for
             distributed computation (see `NVIDIA cuSOLVER tile strategy
             <https://docs.nvidia.com/cuda/cusolver/index.html#tile-strategy>`_).
             Defaults to ``A.shape[0] // jax.local_device_count()`` to distribute
             work evenly across devices, with a minimum of 64.

             The tile size determines the memory/communication tradeoff:

             - **Larger tiles** (closer to matrix size): Less communication overhead,
               more memory per device
             - **Smaller tiles** (64-1024): More communication, less memory per device,
               better load balancing

             For most applications, the default (matrix size / device count) works well.
             Adjust if you encounter memory issues (decrease local_tile_size) or excessive
             communication overhead (increase local_tile_size).
        x0: unused (kept for API compatibility)

    Returns:
        tuple: (solution, None) where solution is the unraveled result.

    Example:
        >>> import netket as nk
        >>> import jax
        >>> from jax.sharding import PartitionSpec as P
        >>>
        >>> # For multi-GPU setup with sharding enabled
        >>> solver = nk.optimizer.solver.cholesky_distributed(local_tile_size=2**12)
        >>> driver = nk.driver.VMC_SR(
        ...     hamiltonian, optimizer, variational_state=vstate,
        ...     linear_solver=solver, diag_shift=0.01
        ... )
    """
    del x0

    jaxmg = import_optional_dependency("jaxmg", descr="cholesky_distributed solver")
    from jax.sharding import PartitionSpec as P

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    # JAXMg's potrs expects b to be 2D
    if b.ndim == 1:
        b = jnp.expand_dims(b, axis=1)
        squeeze_output = True
    else:
        squeeze_output = False

    # Set default tile size: distribute matrix evenly across devices
    if local_tile_size is None:
        n_devices = jax.local_device_count()
        # Divide matrix among devices, with minimum tile size of 64
        local_tile_size = max(64, A.shape[0] // n_devices)
        # Clamp to matrix size
        local_tile_size = min(local_tile_size, A.shape[0])

    mesh = jax.sharding.get_abstract_mesh()
    in_specs = (P("S", None), P(None, None))
    x = jaxmg.potrs(A, b, T_A=local_tile_size, mesh=mesh, in_specs=in_specs)

    if squeeze_output:
        x = jnp.squeeze(x, axis=1)

    return unravel(x), None


@reference(
    "Wiersema2026jaxmg",
    condition="If using pinv_smooth_distributed solver",
    message="This work used the JAXMg distributed linear solver described in Ref.",
)
@partial_from_kwargs
def pinv_smooth_distributed(
    A,
    b,
    *,
    rtol: float = 1e-14,
    rtol_smooth: float = 1e-14,
    local_tile_size=None,
    x0=None,
    rcond: float = None,
    rcond_smooth: float = None,
):
    r"""
    Solve the linear system using a distributed pseudo-inverse from eigendecomposition.

    This is the distributed/multi-GPU equivalent of :func:`~netket.optimizer.solver.pinv_smooth`.
    It uses jaxmg's `syevd` function (cuSOLVERMg SYEVD) to compute eigenvalues and eigenvectors
    of a symmetric/Hermitian matrix in a distributed manner, then applies the same smoothed
    regularization as :func:`~netket.optimizer.solver.pinv_smooth`.

    The eigenvalues :math:`\lambda_i` smaller than
    :math:`r_\textrm{tol} \lambda_\textrm{max}` are truncated (where
    :math:`\lambda_\textrm{max}` is the largest eigenvalue).

    The eigenvalues are further smoothed with another filter, originally introduced in
    `Medvidovic, Sels arXiv:2212.11289 (2022) <https://arxiv.org/abs/2212.11289>`_,
    given by the following equation

    .. math::

        \tilde\lambda_i^{-1}=\frac{\lambda_i^{-1}}{1+\big(\epsilon\frac{\lambda_\textrm{max}}{\lambda_i}\big)^6}


    .. note::

        This solver requires the optional `jaxmg` package to be installed:

        .. code-block:: bash

            pip install jaxmg

        If jaxmg is not installed, an ImportError will be raised.

    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.

    .. note::

        This solver is specifically designed for use with
        `config.netket_experimental_sharding = True` and expects the input
        matrix A to be sharded with `P("S", None)`.

        For single-device or small-scale computations, use
        :func:`~netket.optimizer.solver.pinv_smooth` instead, which has less
        overhead. Use `pinv_smooth_distributed` when:

        - Running on multiple GPUs with sharded arrays
        - NTK/QGT matrix is very large and doesn't fit on a single device
        - You need to minimize communication in distributed settings

    Args:
        A: the matrix A in Ax=b (should be symmetric/Hermitian, sharded)
        b: the vector b in Ax=b
        rtol: Relative tolerance for small eigenvalues of :code:`A`. For
            the purposes of rank determination, eigenvalues are treated
            as zero if they are smaller than `rtol` times the largest
            eigenvalue of :code:`A`.
        rtol_smooth: Regularization parameter used with a similar effect to `rtol`
            but with a softer curve. See :math:`\epsilon` in the formula
            above.
        local_tile_size: Tile size for matrix A. Controls the tiling strategy for
             distributed computation (see `NVIDIA cuSOLVER tile strategy
             <https://docs.nvidia.com/cuda/cusolver/index.html#tile-strategy>`_).
             Defaults to ``A.shape[0] // jax.local_device_count()`` to distribute
             work evenly across devices, with a minimum of 64.

             The tile size determines the memory/communication tradeoff:

             - **Larger tiles** (closer to matrix size): Less communication overhead,
               more memory per device
             - **Smaller tiles** (64-1024): More communication, less memory per device,
               better load balancing

             For most applications, the default (matrix size / device count) works well.
             Adjust if you encounter memory issues (decrease local_tile_size) or excessive
             communication overhead (increase local_tile_size).
        rcond: (deprecated) Alias for `rtol`. Will be removed in a future release.
        rcond_smooth: (deprecated) Alias for `rtol_smooth`. Will be removed in a future release.
        x0: unused (kept for API compatibility)

    Returns:
        tuple: (solution, None) where solution is the unraveled result.

    Example:
        >>> import netket as nk
        >>> import jax
        >>> from jax.sharding import PartitionSpec as P
        >>>
        >>> # For multi-GPU setup with sharding enabled
        >>> solver = nk.optimizer.solver.pinv_smooth_distributed(
        ...     rtol=1e-14, rtol_smooth=1e-14, local_tile_size=2**12
        ... )
        >>> driver = nk.driver.VMC_SR(
        ...     hamiltonian, optimizer, variational_state=vstate,
        ...     linear_solver=solver, diag_shift=0.01
        ... )
    """
    del x0

    if rcond is not None:
        warn_deprecation(
            "The 'rcond' argument is deprecated and will be removed in a future release. "
            "Please use 'rtol' instead."
        )
        rtol = rcond

    if rcond_smooth is not None:
        warn_deprecation(
            "The 'rcond_smooth' argument is deprecated and will be removed in a future release. "
            "Please use 'rtol_smooth' instead."
        )
        rtol_smooth = rcond_smooth

    jaxmg = import_optional_dependency("jaxmg", descr="pinv_smooth_distributed solver")
    from jax.sharding import PartitionSpec as P

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)

    # Set default tile size: distribute matrix evenly across devices
    if local_tile_size is None:
        n_devices = jax.local_device_count()
        # Divide matrix among devices, with minimum tile size of 64
        local_tile_size = max(64, A.shape[0] // n_devices)
        # Clamp to matrix size
        local_tile_size = min(local_tile_size, A.shape[0])

    mesh = jax.sharding.get_abstract_mesh()
    in_specs = (P("S", None),)

    # Compute eigendecomposition using distributed solver
    Σ, U = jaxmg.syevd(A, T_A=local_tile_size, mesh=mesh, in_specs=in_specs)

    # Discard eigenvalues below numerical precision
    Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rtol, jnp.reciprocal(Σ), 0.0)

    # Set regularizer for singular value cutoff
    regularizer = 1.0 / (1.0 + (rtol_smooth / jnp.abs(Σ / Σ[-1])) ** 6)

    Σ_inv = Σ_inv * regularizer

    x = U @ (Σ_inv * (U.conj().T @ b))

    return unravel(x), None
