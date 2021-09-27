from flax import struct
from numbers import Number

from .core import PyTreeArray, PyTreeArray1

import jax

@struct.dataclass
class _IndexUpdateHelper:
    pytreearr : PyTreeArray

    def __getitem__(self, indices):
        return _IndexUpdateRef(self.pytreearr, indices)

@struct.dataclass
class _IndexUpdateRef:    
    pytreearr : PyTreeArray
    index: "idx"

    def set(self, val):

        if isinstance(val, Number) or hasattr(val, "shape") and val.ndim == 0:
            def _update(x):
                return x.at[self.index].set(val)
            return jax.tree_map(_update, self.pytreearr)

        else:
            def _update(x,val):
                return x.at[self.index].set(val)
            _val = val if not isinstance(val, PyTreeArray) else val.tree
            _tree = jax.tree_multimap(_update, self.pytreearr.tree, _val)
            return PyTreeArray(_tree, self.pytreearr.treedefs, self.pytreearr.axes)
