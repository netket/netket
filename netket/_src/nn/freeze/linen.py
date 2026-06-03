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

from typing import Any, Callable

from flax import linen as nn

from netket._src.nn.freeze.common import freeze_variables, merge_params


class FreezeLinenWrapper(nn.Module):
    """
    Transparent Flax Linen wrapper that supports parameter freezing.

    After initializing a variational state with this wrapper, call the
    high-level :func:`~netket.nn.freeze.freeze_parameters` to move a subset of
    parameters from the trainable ``"params"`` collection into the
    ``"frozen_params"`` collection.  NetKet treats ``"frozen_params"`` as
    non-trainable model state, so the frozen parameters are excluded from
    gradient computation and optimizer updates automatically.

    The wrapper uses :func:`flax.linen.share_scope` so the inner model's
    variables stay at the top level — the variable tree is identical to the
    bare model's, plus the extra ``"frozen_params"`` collection.  At apply time
    the trainable and frozen parameters are merged back transparently before
    delegating to the inner model.

    Create instances via :meth:`from_module_and_variables` rather than calling
    the constructor directly when you already have a set of variables.

    Note:
        For NNX modules, prefer :func:`~netket.nn.freeze.freeze_params` which
        mutates the module in-place without requiring a wrapper.
    """

    model: nn.Module

    def setup(self):
        # Share the inner model's scope so its variables (``"params"``,
        # ``"batch_stats"``, ...) appear at the top level instead of being
        # nested under a ``"model"`` sub-key.
        nn.share_scope(self, self.model)

    @classmethod
    def from_module_and_variables(
        cls,
        bare_module: "nn.Module",
        bare_variables: dict,
        is_frozen: Callable[[tuple[str, ...], Any], bool] | None = None,
    ) -> tuple["FreezeLinenWrapper", dict]:
        """
        Create a :class:`FreezeLinenWrapper` from an unbound module and variables.

        Because the wrapper shares the inner model's scope, *bare_variables* are
        used as-is (no re-nesting).  An empty ``"frozen_params"`` collection is
        added, and if *is_frozen* is given the ``"params"`` collection is
        immediately split into trainable and frozen subsets.

        Args:
            bare_module: The unbound Linen module to wrap.
            bare_variables: Variables dict from the bare module (e.g. the result
                of ``bare_module.init(...)`` or ``vstate.variables``).
            is_frozen: Optional callable ``(path, leaf) -> bool``.  If provided,
                parameters for which this returns ``True`` are frozen immediately.

        Returns:
            ``(wrapper, new_variables)``
        """
        wrapper = cls(model=bare_module)

        variables = dict(bare_variables)
        variables.setdefault("frozen_params", {})

        if is_frozen is not None:
            variables = freeze_variables(variables, is_frozen)

        return wrapper, variables

    def __call__(self, x, *args, **kwargs):
        if self.is_initializing():
            # Let Linen initialize the inner model normally.  Thanks to the
            # shared scope its collections land at the top level.
            return self.model(x, *args, **kwargs)

        # During apply: merge trainable ("params") and frozen ("frozen_params")
        # parameters back together and forward all other collections untouched.
        merged_params = merge_params(
            self.variables.get("params", {}),
            self.variables.get("frozen_params", {}),
        )
        inner_variables = {"params": merged_params}
        for collection, val in self.variables.items():
            if collection not in ("params", "frozen_params"):
                inner_variables[collection] = val

        if self.scope.mutable is False:
            return self.model.apply(inner_variables, x, *args, **kwargs)

        out, mutable_variables = self.model.apply(
            inner_variables,
            x,
            *args,
            mutable=self.scope.mutable,
            **kwargs,
        )
        for collection, val in mutable_variables.items():
            if collection in ("params", "frozen_params"):
                continue
            for name, leaf in val.items():
                self.put_variable(collection, name, leaf)
        return out

    def unfreeze(self, variables: dict) -> tuple["nn.Module", dict]:
        """
        Recover the original bare module and its variables (inverse of freezing).

        Merges the ``"frozen_params"`` collection back into ``"params"``, drops
        the now-empty ``"frozen_params"`` key, and returns the *unwrapped* inner
        module together with the reconstructed variables.  This fully undoes
        :meth:`from_module_and_variables`: ``self.model.apply(restored, x)``
        reproduces the pre-freeze output.

        Args:
            variables: Variables dict of this wrapper (with ``"frozen_params"``).

        Returns:
            ``(bare_module, bare_variables)`` — *bare_module* is the original
            inner module, *bare_variables* has every parameter back in
            ``"params"`` and no ``"frozen_params"`` collection.
        """
        merged_params = merge_params(
            variables.get("params", {}),
            variables.get("frozen_params", {}),
        )
        bare_variables = {"params": merged_params}
        for collection, val in variables.items():
            if collection not in ("params", "frozen_params"):
                bare_variables[collection] = val
        return self.model, bare_variables
