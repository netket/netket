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
from flax import linen as nn
from typing import Any


class ApplyOperatorModuleLinen(nn.Module):
    """
    A Flax Linen module that wraps another module and applies an operator transformation.

    This module wraps a base neural network module and applies an operator in front of it,
    computing log(O|ψ⟩) where O is the operator and |ψ⟩ is represented by the base module.

    The operator is stored in flattened form. The static structure (treedef) is stored as a
    module attribute, while the dynamic data (leaves) is stored in the 'operator' variable
    collection, which is not trainable. This separation allows the operator's arrays to be
    updated without triggering recompilation while keeping the structure static.

    Args:
        base_module: The Flax module to wrap
        operator_treedef: The pytree structure of the operator (obtained from jax.tree.flatten)

    Example::

        import netket as nk
        hilbert = nk.hilbert.Spin(0.5, 4)
        base_module = nk.models.RBM(alpha=1)
        operator = nk.operator.spin.sigmax(hilbert, 0)

        # Flatten the operator to separate static structure from dynamic data
        leaves, treedef = jax.tree.flatten(operator)
        transformed = ApplyOperatorModuleLinen(base_module=base_module, operator_treedef=treedef)

        # Initialize: first init the base module to get its params
        base_params = base_module.init(jax.random.key(1), hilbert.all_states())
        # Then add only the operator leaves to the variables dict
        variables = {**base_params, 'operator': {'leaves': leaves}}

        # Apply the transformed module
        logpsi = transformed.apply(variables, x)

        # The operator can be updated without recompilation
        # Only update the leaves (treedef is fixed in the module)
        new_leaves, _ = jax.tree.flatten(new_operator)
        variables['operator']['leaves'] = new_leaves
        logpsi = transformed.apply(variables, x)
    """

    base_module: nn.Module
    """The base module to wrap"""

    operator_treedef: Any
    """The static pytree structure of the operator"""

    @classmethod
    def from_module_and_variables(
        cls, bare_module: nn.Module, operator, bare_variables: dict
    ):
        """
        Create a TransformedModule from a bare module, operator, and variables.

        Args:
            bare_module: The bare Flax module to wrap
            operator: The operator to apply (will be flattened)
            bare_variables: The variables dictionary from the bare module

        Returns:
            A tuple of (wrapped_module, wrapped_variables) where:
            - wrapped_module is the new TransformedModule instance
            - wrapped_variables is the properly structured variables dict
        """
        leaves, treedef = jax.tree.flatten(operator)
        wrapped_module = cls(base_module=bare_module, operator_treedef=treedef)

        # Wrap the base module's variables under 'base_module' key to match Flax's scoping
        wrapped_variables = {
            collection: {"base_module": params}
            for collection, params in bare_variables.items()
        }

        # Add the new operator at the top level (outer module's operator)
        if "operator" not in wrapped_variables:
            wrapped_variables["operator"] = {}
        wrapped_variables["operator"]["leaves"] = leaves

        return wrapped_module, wrapped_variables

    @property
    def operator(self):
        """Reconstruct the operator from its flattened representation."""
        if not self.has_variable("operator", "leaves"):
            raise ValueError(
                "Operator leaves not found in variables. "
                "Make sure to add the operator leaves to variables:\n"
                "  leaves, treedef = jax.tree.flatten(operator)\n"
                "  variables['operator'] = {'leaves': leaves}"
            )
        leaves = self.get_variable("operator", "leaves")
        return jax.tree.unflatten(self.operator_treedef, leaves)

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        if self.is_initializing():
            raise ValueError(
                "ApplyOperatorModuleLinen cannot be initialized using the standard init() method. "
                "Instead, initialize the base_module separately and provide operator leaves:\n"
                "  base_params = base_module.init(key, x)\n"
                "  leaves, treedef = jax.tree.flatten(operator)\n"
                "  variables = {**base_params, 'operator': {'leaves': leaves}}"
            )

        # TODO: Move to global import to avoid circular dependencies
        from netket.operator import ContinuousOperator

        # Reconstruct the operator using the property
        stored_operator = self.operator

        if isinstance(stored_operator, ContinuousOperator):
            # For continuous operators, use their specialized kernel
            # Create a wrapper function that calls the base module with apply()
            base_variables = {
                k: v for k, v in self.variables.items() if k != "operator"
            }

            def apply_base(variables, x_in, *args_in, **kwargs_in):
                return self.base_module.apply(variables, x_in, *args_in, **kwargs_in)

            res = stored_operator._expect_kernel(
                apply_base, base_variables, x, *args, **kwargs
            )
        else:
            xp, mels = stored_operator.get_conn_padded(x)
            xp = xp.reshape(-1, x.shape[-1])

            # Call the base module on the connected configurations using apply()
            logpsi_xp = self.base_module(xp, *args, **kwargs)
            logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

            res = jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

        return res
