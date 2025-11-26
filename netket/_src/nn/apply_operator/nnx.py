# Copyright 2025 The NetKet Authors - All rights reserved.
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
from flax import nnx


class ApplyOperatorModuleNNX(nnx.Module):
    """
    A Flax NNX module that wraps another NNX module and applies an operator transformation.

    This module wraps a base neural network module and applies an operator in front of it,
    computing log(O|ψ⟩) where O is the operator and |ψ⟩ is represented by the base module.

    The operator is stored as a regular NNX Variable with collection='operator', which makes
    it non-trainable by default (since optimizers typically only update 'params' collection).

    Unlike the Linen version, NNX modules are stateful and contain their parameters directly.
    This makes the implementation more straightforward - we just store the base module and
    operator as attributes.

    Args:
        base_module: The NNX module to wrap
        operator: The operator to apply

    Example::

        import netket as nk
        from netket.models import RBM
        from flax import nnx

        # Create base NNX module (already initialized with parameters)
        base_module = RBM(N=10, alpha=2, rngs=nnx.Rngs(0))
        operator = nk.operator.spin.sigmax(hilbert, 0)

        # Create transformed module
        transformed = ApplyOperatorModuleNNX(base_module, operator)

        # Use it directly (NNX style)
        logpsi = transformed(x)

        # Or use with MCState
        vstate = nk.vqs.MCState(sampler, transformed, n_samples=1000)

        # The operator can be updated
        transformed.operator = new_operator
        logpsi = transformed(x)
    """

    def __init__(self, base_module: nnx.Module, operator):
        """
        Initialize the transformed module.

        Args:
            base_module: The base NNX module to wrap
            operator: The operator to apply in front of the base module
        """
        self.base_module = base_module

        # Cannot store directly the operator, as it would break the
        # assumption that variables are only pure dictionaries.
        leaves, treedef = jax.tree.flatten(operator)
        self._operator_treedef = treedef
        self._operator_leaves = nnx.Variable(leaves, collection="operator")

    @property
    def operator(self):
        """The operator applied to the base module."""
        return jax.tree.unflatten(
            self._operator_treedef, self._operator_leaves.get_value()
        )

    @operator.setter
    def operator(self, new_operator):
        leaves, treedef = jax.tree.flatten(new_operator)
        self._operator_treedef = treedef
        self._operator_leaves.set_value(leaves)

    def __nnx_repr__(self):
        """Custom repr to avoid issues with operator repr in NNX."""
        from flax.nnx import reprlib

        yield reprlib.Object(type(self))
        yield reprlib.Attr("base_module", self.base_module)
        yield reprlib.Attr("operator", self.operator)

    def __call__(self, x, *args, **kwargs):
        from netket.operator import ContinuousOperator

        # Get the operator (reconstructed from flattened representation)
        stored_operator = self.operator

        # Apply the operator transformation
        if isinstance(stored_operator, ContinuousOperator):
            # For continuous operators, we need to create a wrapper function
            # that extracts and recomposes the base module for the kernel
            def apply_base(variables, x_in, *args_in, **kwargs_in):
                # In NNX, we need to split and merge the base module
                graphdef, state = nnx.split(self.base_module)
                base_module_copy = nnx.merge(graphdef, variables)
                return base_module_copy(x_in, *args_in, **kwargs_in)

            # Split the base module to get its state as variables
            _, base_variables = nnx.split(self.base_module)

            res = stored_operator._expect_kernel(
                apply_base, base_variables, x, *args, **kwargs
            )
        else:
            # For discrete operators, use the standard approach
            xp, mels = stored_operator.get_conn_padded(x)
            xp = xp.reshape(-1, x.shape[-1])

            # Call the base module directly (NNX modules are stateful)
            logpsi_xp = self.base_module(xp, *args, **kwargs)
            logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

            # Compute log(sum(mels * exp(logpsi_xp)))
            res = jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

        return res
