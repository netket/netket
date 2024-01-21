# Copyright 2023 The NetKet Authors - All rights reserved.
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

# Use an empty top-level docstring so Sphinx won't output the one below.
""""""

from textwrap import dedent as _dedent

"""NetKet error classes.
(Inspired by NetKet error classes)

=== When to create a NetKet error class?

If an error message requires more explanation than a one-liner, it is useful to
add it as a separate error class. This may lead to some duplication with
existing documentation or docstrings, but it will provide users with more help
when they are debugging a problem. We can also point to existing documentation
from the error docstring directly.

=== How to name the error class?

* If the error occurs when doing something, name the error
  <Verb><Object><TypeOfError>Error

  For instance, if you want to raise an error when applying a module with an
  invalid method, the error can be: ApplyModuleInvalidMethodError.

 <TypeOfError> is optional, for instance if there is only one error when
  modifying a variable, the error can simply be: ModifyVariableError.

* If there is no concrete action involved the only a description of the error is
  sufficient. For instance: InvalidFilterError, NameInUseError, etc.


=== Copy/pastable template for new error messages:

class Template(NetketError):
  "" "

  "" "
  def __init__(self):
    super().__init__(f'')
"""


class NetketError(Exception):
    def __init__(self, message):
        error_index = "https://netket.readthedocs.io/en/latest/api/errors.html"
        error_dir = "https://netket.readthedocs.io/en/latest/api/_generated/errors"
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        error_msg = (
            f"{_dedent(message)}"
            f"\n"
            f"\n-------------------------------------------------------"
            f"\n"
            f"For more detailed informations, visit the following link:"
            f"\n\t {error_dir}/{module_name}.{class_name}.html"
            f"\n"
            f"or the list of all common errors at"
            f"\n\t {error_index}"
            f"\n-------------------------------------------------------"
            f"\n"
        )
        super().__init__(error_msg)


class NetketWarning(Warning):
    def __init__(self, message):
        error_index = "https://netket.readthedocs.io/en/latest/api/errors.html"
        error_dir = "https://netket.readthedocs.io/en/latest/api/_generated/errors"
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__

        error_msg = (
            f"{_dedent(message)}"
            f"\n"
            f"\n-------------------------------------------------------"
            f"\n"
            f"For more detailed informations, visit the following link:"
            f"\n\t {error_dir}/{module_name}.{class_name}.html"
            f"\n"
            f"or the list of all common errors and warnings at"
            f"\n\t {error_index}"
            f"\n-------------------------------------------------------"
            f"\n"
        )
        super().__init__(error_msg)


#################################################
# Hilbert errors                                #
#################################################


class HilbertIndexingDuringTracingError(NetketError):
    """Illegal attempt to use state indexing function from
    inside of a Jax function transformation.

    This happens when you call functions such as
    :meth:`~netket.hilbert.DiscreteHilbert.states_to_numbers` or
    its opposite :meth:`~netket.hilbert.DiscreteHilbert.numbers_to_states` from
    within a :func:`jax.jit`, :func:`jax.grad`, :func:`jax.vjp` or similar jax
    function transformations.

    There is currently no workaround rather than returning the arrays from
    the jax function and performing the conversion outside of it.

    """

    def __init__(self):
        super().__init__(
            "\n"
            "Attempted to use state-indexing functions of an hilbert object "
            "inside a Jax function transformation (jax.jit, jax.grad & others)."
            "\n\n"
            "Functions that convert states to indices and vice-versa such as "
            "`hilbert.states_to_numbers()` or `hilbert.numbers_to_states()` "
            "are implemented in Numba and cannot be executed from within a"
            "jax function transformation."
        )


#################################################
# Operators errors                              #
#################################################


class NumbaOperatorGetConnDuringTracingError(NetketError):
    """Illegal attempt to use Numba-operators inside of a Jax
    function transformation.

    This happens when calling :meth:`~netket.operator.DiscreteOperator.get_conn_padded` or
    :meth:`~netket.operator.DiscreteOperator.get_conn_flattened` inside of a function
    that is being transformed by jax with transformations such as :func:`jax.jit`
    or :func:`jax.grad`, and the operator is not compatible with Jax.

    To avoid this error you can (i) convert your operator to a Jax compatible format if possible,
    or (ii) compute the connected elements outside of the jax function transformation and pass the
    results to a jax-transformed function.

    (i) Converting an Operator to a Jax compatible format
    -----------------------------------------------------

    Some operators can be converted to a jax-compatible format by calling the method
    `operator.to_jax_operator()`. If this method is not available or raises an error, it means
    that the operator cannot be converted.

    If the operator can be converted to a jax-compatible format, it will be possible to pass
    it as a standard argument to a jax-transformed function and it should not be declared as
    a static argument.

    Jax compatible operators can be used like standard operators, for example by passing
    it to :meth:`~netket.vqs.MCState.expect` function. However, the performance will differ from
    standard operators. In general, you might find that compile time will be much worse, while
    runtime might be faster or slower, depending on several factors.

    The biggest advantage to Jax operators, however, is when experimenting with jax code, as you
    can succesfully use them in your own custom functions as in the example below:

    .. code-block:: python

        import netket as nk
        import jax
        import jax.numpy as jnp

        graph = nk.graph.Chain(10)
        hilbert = nk.hilbert.Spin(1/2, graph.n_nodes)
        ham = nk.operator.Ising(hilbert, graph, h=1.0)
        ham_jax = ham.to_jax_operator()

        ma = nk.models.RBM()
        pars = ma.init(jax.random.PRNGKey(1), jnp.ones((2,graph.n_nodes)))

        samples = hilbert.all_states()

        @jax.jit
        def compute_local_energies(pars, ham_jax, s):
            # this would raise the error
            sp, mels = ham.get_conn_padded(s)
            # this will work
            sp, mels = ham_jax.get_conn_padded(s)

            logpsi_sp = ma.apply(pars, sp)
            logpsi_s = jnp.expand_dims(ma.apply(pars, s), -1)

            return jnp.sum(mels * jnp.exp(logpsi_sp-logpsi_s), axis=-1)

        elocs = compute_local_energies(pars, ham_jax, samples)
        elocs_grad = jax.jacrev(compute_local_energies)(pars, ham_jax, samples)

    .. note::

        Note that this transformation might be a relatively expensive operation, so you should avoid
        executing this inside of an hot loop.


    (ii) Precomputing connected elements outside of Jax transformations
    -------------------------------------------------------------------

    In most cases you won't be able to convert the operator to a Jax-compatible format.
    In those cases, the workaround we usually employ is to precompute the connected elements
    before entering the Jax context, splitting our function into a non-jitted function and
    into a jitted kernel.

    .. code-block:: python

      import netket as nk
      import jax
      import jax.numpy as jnp

      graph = nk.graph.Chain(10)
      hilbert = nk.hilbert.Spin(1/2, graph.n_nodes)
      ham = nk.operator.Ising(hilbert, graph, h=1.0)

      ma = nk.models.RBM()
      pars = ma.init(jax.random.PRNGKey(1), jnp.ones((2,graph.n_nodes)))

      samples = hilbert.all_states()

      def compute_local_energies(pars, ham, s):
          sp, mels = ham.get_conn_padded(s)
          return _compute_local_energies_kernel(pars, s, sp, mels)

      @jax.jit
      def _compute_local_energies_kernel(pars, s, sp, mels):
          logpsi_sp = ma.apply(pars, sp)
          logpsi_s = jnp.expand_dims(ma.apply(pars, s), -1)
          return jnp.sum(mels * jnp.exp(logpsi_sp-logpsi_s), axis=-1)

      elocs = compute_local_energies(pars, ham_jax, samples)
      elocs_grad = jax.jacrev(compute_local_energies)(pars, ham_jax, samples)


    Most :class:`~netket.operator.DiscreteOperator`s are implemented in Numba and therefore are not jax-compatible.
    To know if it's valid to use an operator inside of a jax function transformation you can
    check that it inherits from :class:`~netket.operator.DiscreteJaxOperator` by executing

    .. code-block:: python

        isinstance(operator, nk.operator.DiscreteJaxOperator)

    """

    def __init__(self, operator):
        operator_type = type(operator)
        super().__init__(
            f"\n"
            f"Attempted to use a Numba-based operator ({operator_type}) "
            f"inside a Jax function transformation (jax.jit, jax.grad & others)."
            f"\n\n"
            f"Numba-based operators are not compatible with Jax function "
            f"transformations, and can only be used outside of jax-function "
            f"boundaries."
            f"\n\n"
            f"Some operators can be converted to a Jax-compatible version by"
            f"calling `operator.to_jax_operator()`, but not all support it."
            f"If your operator does not support it, read the documentation to "
            f"find how to work-around this issue."
        )


class JaxOperatorSetupDuringTracingError(NetketError):
    """Illegal attempt to use a Jax-operator Numba-operators constructed inside of a Jax
    function transformation with non-constant data.

    This happens when building a :class:`~netket.operator.DiscreteJaxOperator` inside
    of a function that is being transformed by jax with transformations such as :func:`jax.jit`
    or :func:`jax.grad`, and the operator is not compatible with Jax.

    Notice that :class:`~netket.operator.DiscreteJaxOperator` can be used inside of jax
    function transformations, but NetKet is currently limited in that you cannot build
    them inside of Jax transformations.

    To avoid this error you should build your operators outside of the jax context.

    (i) Building a Jax operator outside of a jax context
    ----------------------------------------------------

    Build the operator outside of a jax context.

    .. code-block:: python

        import netket as nk
        import jax
        import jax.numpy as jnp

        N = 2

        ham = nk.operator.PauliStringsJax(['XI', 'IX'], jnp.array([0.3, 0.4]))

        samples = ham.hilbert.all_states()

        @jax.jit
        def compute_values(ham, s):
            return ham.get_conn_padded(s)

        compute_values(ham, samples)


    .. note::

        This limitation is not systematic, and it could be lifted in the future by some
        interested coder. If at that moment Jax will support dynamic shape, this feature could
        be implemented at no additional runtime cost. If Jax won't support yet dynamic shapes
        then it should be implemented as a secondary path (instead of this error) that is only
        taken if the operator is constructed inside of a jax context. This path will lead to a
        slightly less optimized, higher computational cost operators.

        If you are really interested in contributing to NetKet and find yourself in need of
        building operators in a jax context (for example because you're doing optimal control),
        get in touch with us by opening an issue.

    .. note::

        Most operators lazily initialise the fields used to compute the connected elements only
        when needed. To check whether an operator was initialized you can probe the boolean flag
        :code:`operator._initialized`. If :code:`operator._initialized` is True, you can safely call
        :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded` and similar methods. If it is
        False, then the setup procedure will be handled by an internal method usually called
        :code:`operator._setup()`. If you see this error, it means that this method internally
        uses dynamically determined shapes, and it is what should be converted to be jax-friendly.


    """

    def __init__(self, operator):
        operator_type = type(operator)
        super().__init__(
            f"\n"
            f"Attempted to initialise a jax operator ({operator_type}) "
            f"inside a Jax function transformation (jax.jit, jax.grad & others)."
            f"\n\n"
            f"Jax-based operators cannot be initialised (constructed) within Jax "
            f"function transformations, and can only be initialised outside Jax "
            f"function boundaries."
            f"\n\n"
            f"To avoid this error, construct the operator outside of the jax"
            f"function and pass it to it as a standard argument."
        )


class JaxOperatorNotConvertibleToNumba(NetketError):
    """Illegal attempt to convert to the Numba format a Jax operator that had been flattened
    and unflattened.

    This probably happened because you passed a Jax operator to a jax function transformation
    or jitted function and then tried to re-convert it to the numba format like in the example
    below:

    .. code-block:: python

        import netket as nk

        hi = nk.hilbert.Spin(0.5, 2)

        op = nk.operator.spin.sigmax(hi, 0)
        op = op.to_jax_operator()

        @jax.jit
        def test(op):
            op.to_numba_operator()

        test(op)

    Unfortunately, once an operator is flattened with {ref}`jax.tree_util.tree_flatten`, which
    happens at all jax-function transformation boundaries, it usually cannot be converted back to
    the original numba form.

    This happens for performance reasons, and we might reconsider. If it is a problem for you, do
    open an issue.

    """

    def __init__(self, operator):
        super().__init__(
            "\n"
            "Illegal attempt to convert to the Numba format a Jax operator that had been flattened "
            "and unflattened."
            "\n\n"
            "Jax-based operators cannot be flattened or passed to a jax function and then be "
            "converted to their numba format."
        )


#################################################
# Jacobian and QGT errors                       #
#################################################


class IllegalHolomorphicDeclarationForRealParametersError(NetketError):
    """Illegal attempt to declare a function holomorphic when it has some
    or all real parameters, which makes it automatically non-holomorphic.

    This error may arise when computing the Jacobian directly or when
    constructing the Quantum Geometric Tensor, which internally constructs
    the Jacobian of your variational function.

    In general we could silence this error automatically and ignore the
    :code:`holomorphic=True` argument, but we wish to stress with the users
    the importance of correctly declaring this argument.

    To solve this error, simply stop declaring :code:`holomorphic=True`, by
    either removing it or specifying :code:`holomorphic=False`.
    """

    def __init__(self):
        super().__init__(
            """
        A function with real parameters is not holomorphic.

        You declared `holomorphic=True` when computing the Jacobian, Quantum
        Geometric Tensor or a similar, but your variational function has
        real parameters, so it cannot be holomorphic.

        To fix this error, remove the keyword argument `holomorphic=True`.
        """
        )


class NonHolomorphicQGTOnTheFlyDenseRepresentationError(NetketError):
    """
    QGTOnTheFly cannot be converted to a dense matrix for non-holomorphic
    functions which have complex parameters.

    This limitation does not apply if the parameters are all real.

    This error might have happened for two reasons:
     - you specified `holomorphic=False` because your ansatz is non-holomorphic.
       In that case you should use `QGTJacobianPyTree` or `QGTJacobianDense`
       implementations.

     - you did not specify `holomorphic`, which leads to the default value of
       `holomorphic=False` (in that case, you should have seen a warning). If
       that is the case, you should carefully check if your ansatz is holomorhic
       almost everywhere, and if that is the case, specify `holomorphic=True`.
       If your ansatz is not-holomorphic, the same suggestion as above applies.

    Be warned that if you specify `holomorphic=True` when your ansatz is mathematically
    not holomorphic is a surprisingly bad idea and will lead to numerically wrong results,
    so I'd invite you not to lie to your computer.
    """

    def __init__(self):
        super().__init__(
            """
            QGTOnTheFly cannot be converted to a dense matrix for non-holomorphic
            functions which have complex parameters.

            This limitation does not apply if the parameters are all real.

            This error might have happened for two reasons:
             - you specified `holomorphic=False` because your ansatz is non-holomorphic.
               In that case you should use `QGTJacobianPyTree` or `QGTJacobianDense`
               implementations.

             - you did not specify `holomorphic`, which leads to the default value of
               `holomorphic=False` (in that case, you should have seen a warning). If
               that is the case, you should carefully check if your ansatz is holomorhic
               almost everywhere, and if that is the case, specify `holomorphic=True`.
               If your ansatz is not-holomorphic, the same suggestion as above applies.

            Be warned that if you specify `holomorphic=True` when your ansatz is mathematically
            not holomorphic is a surprisingly bad idea and will lead to numerically wrong results,
            so I'd invite you not to lie to your computer.
            """
        )


class HolomorphicUndeclaredWarning(NetketWarning):
    """
    Complex-to-Complex model detected. Defaulting to :code:`holomorphic = False`
    for the calculation of its jacobian.

    However, :code:`holomorphic = False` might lead to slightly increased
    computational cost, some disabled features and/or worse quality of
    solutions found with iterative solvers.
    If your model is actually holomorphic, you should specify :code:`holomorphic = True`
    to unblock some extra, possibly more performant algorithms.

    If you are unsure whether your variational function is holomorphic or not,
    you should check if it satisfies the
    `Cauchy-Riemann equations <https://en.wikipedia.org/wiki/Cauchy–Riemann_equations>`_.

    To check numerically those conditions on a random set of samples you can use the
    function :func:`netket.utils.is_probably_holomorphic`. If this function returns
    False then your ansatz is surely not holomorphic, while if it returns True your
    ansatz is likely but not guaranteed to be holomorphic.

    To check those conditions numerically, you can check by following this
    example:

    .. code:: python

        hi = nk.hilbert.Spin(0.5, 2)
        sa = nk.sampler.MetropolisLocal(hi)
        ma = nk.models.RBM(param_dtype=complex)
        # construct the variational state
        vs = nk.vqs.MCState(sa, ma)

        nk.utils.is_probably_holomorphic(vs._apply_fun,
                                         vs.parameters,
                                         vs.samples,
                                         model_state = vs.model_state)

    To suppress this warning correctly specify the keyword argument
    `holomorphic`.

    .. note::
        a  detailed discussion, explaining how to easily check those conditions
        analitically is found in the documentation of
        :func:`netket.utils.is_probably_holomorphic`).

    """

    def __init__(self):
        super().__init__(
            """
            Defaulting to `holomorphic=False`, but this might lead to increased
            computational cost or disabled features. Check if your variational
            function is holomorphic, and if so specify `holomorphic=True`as an extra
            keyword argument.

            To silence this warning, specify the `holomorphic=False/True` keyword
            argument.

            To numerically check whether your variational function is or not holomorphic
            you can use the following snippet:

            ```python
               vs = nk.vqs.MCState(...)

               nk.utils.is_probably_holomorphic(vs._apply_fun, vs.parameters, vs.samples, vs.model_state)
            ```

            if `nk.utils.is_probably_holomorphic` returns False, then your function is not holomorphic.
            If it returns True, it is probably holomorphic.
            """
        )


class RealQGTComplexDomainError(NetketError):
    """
    This error is raised when you apply the Quantum Geometric Tensor of a
    non-holomorphic function to a complex-valued vector, because the
    operation is not well-defined.

    As is explained in the documentation of the geomtric tensors, the QGT
    implementation for non-holomorphic functions corresponds to the real-
    part of the QGT, not the full QGT.

    This is because in most applications of variational monte carlo, such as
    the Time-Dependent Variational Principle, Ground-state search or
    supervised-learning, you only require knowledge of the real part, and
    can safely discard the imaginary part that would incur in an increased
    computational cost.

    While the product of the real part of QGT by a complex vector is well-defined,
    to prevent the common mistake of assuming that the QGT is complex we
    explicitly raise this error, forcing users to manually multiply the QGT
    by the real and imaginary part of the vector, as we would have to do
    inside of this class anyway.

    If this is really the mathematical operation you want to perform, then
    you can do it manually, but very often we have found that when you
    apply the real part of the QGT to a complex vector you might have
    your math wrong. In such cases, sometimes what you actually wanted
    to do was

    .. code:: python

       >>> import netket as nk; import jax
       >>>
       >>> vstate = nk.vqs.FullSumState(nk.hilbert.Spin(0.5, 5), \
                                        nk.models.RBM(param_dtype=complex))
       >>> _, vec = vstate.expect_and_grad(nk.operator.spin.sigmax(vstate.hilbert, 1))
       >>> G = nk.optimizer.qgt.QGTOnTheFly(vstate, holomorphic=False)
       >>>
       >>> vec_real = jax.tree_map(lambda x: x.real, vec)
       >>> sol = G@vec_real

   Or, if you used the QGT in a linear solver, try using:

   .. code:: python

       >>> import netket as nk; import jax
       >>>
       >>> vstate = nk.vqs.FullSumState(nk.hilbert.Spin(0.5, 5), \
                                        nk.models.RBM(param_dtype=complex))
       >>> _, vec = vstate.expect_and_grad(nk.operator.spin.sigmax(vstate.hilbert, 1))
       >>>
       >>> G = nk.optimizer.qgt.QGTOnTheFly(vstate, holomorphic=False)
       >>> vec_real = jax.tree_map(lambda x: x.real, vec)
       >>>
       >>> linear_solver = jax.scipy.sparse.linalg.cg
       >>> solution, info = G.solve(linear_solver, vec_real)

    """

    def __init__(self):
        super().__init__(
            """
            Cannot multiply the (real part of the) QGT by a complex vector.
            You should either take the real part of the vector, or perform
            the multiplication against the real and imaginary part of the
            vector separately and then recomposing the two.

            This is happening because you have real parameters or a non-holomorphic
            complex wave function. In this case, the Quantum Geometric Tensor object
            only stores the real part of the QGT.

            If you were executing a matmul `G@vec`, try using:

            .. code:: python

               >>> vec_real = jax.tree_map(lambda x: x.real, vec)
               >>> G@vec_real

            If you used the QGT in a linear solver, try using:

            .. code:: python

               >>> vec_real = jax.tree_map(lambda x: x.real, vec)
               >>> G.solve(linear_solver, vec_real)

            to fix this error.

            Be careful whether you need the real or imaginary part
            of the vector in your equations!
            """
        )


class SymmModuleInvalidInputShape(NetketError):
    """
    This error when you attempt to use a module with an input having a wrong number
    of dimensions.

    In particular, Simmetric layers require inputs with 3 dimensions :math:`(B, C, L)`:
        - Batch dimension, which should be 1 if only 1 sample is considered
        - Channel or Features dimension, which should encode multiple features, usually
            originated from previous layers. If this is the first simmetric layer, you
            can set this dimension to 1
        - Length, which should span the physical degrees of freedom.
    """

    def __init__(self, name, x):
        super().__init__(
            """
            Input to DenseSymmFFT has {x.ndim =} but 3 are required.

            The input format is (B,C,L), aka (batches, features/channels, length). If
            you have a single sample, simply use `jnp.atleast_3d(x)` before calling
            this module.

            If this is the first layer in a network, you usually need to set the channel
            dimension to 1.
            """
        )


class UnoptimalSRtWarning(NetketWarning):
    """
    SRt should be used when the number of parameters exceed the number of samples.
    If this is not the case, employing `netket.driver.VMC` with the `nk.optimizer.SR` preconditioner
    is a more efficient option while maintaining the same parameter dynamics.

    .. note::
        a  detailed discussion can be found in the documentation of
        :func:`netket.experimental.driver.VMC_SRt`).

    """

    def __init__(self, n_parameters, n_samples):
        super().__init__(
            f"""
            You are in the case n_samples > num_params ({n_samples} > {n_parameters}),
            for which the `VMC_SRt` is not optimal. Consider using `netket.driver.VMC`
            with the preconditioner `nk.optimizer.SR` to achieve the same parameter dynamics,
            but with improved speed.
            """
        )


#################################################
# Functions to throw errors                     #
#################################################


def concrete_or_error(force, value, error_class, *args, **kwargs):
    """
    Wraps `jax.core.concrete_or_error` but allows us to throw our
    own errors.

    Args:
      force: function to be executed on the value (for example np.asarray)
        that would fail if value is not concrete
      value: the argument to `force`
      error: the constructor of the custom error to throw.
      *args: any additional argument and keyword argument to pass to the custom
        error type constructor.
    """
    import jax

    from jax.core import ConcretizationTypeError

    try:
        return jax.core.concrete_or_error(
            force,
            value,
            """
          """,
        )
    except ConcretizationTypeError as err:
        raise error_class(*args, **kwargs) from err
