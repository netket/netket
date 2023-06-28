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


"""NetKet error classes.
(Inspired by Flax error classes)

=== When to create a Flax error class?

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

class Template(FlaxError):
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
            f"{message}"
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
