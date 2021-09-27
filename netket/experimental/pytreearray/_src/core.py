import jax
import jax.numpy as jnp
import jax.flatten_util
from flax import struct
from functools import partial, reduce
from typing import Any

from operator import mul

from .util import _arr_treedef, amap, _treedefs_compose

from .transpose import transpose
from .dense import to_dense, _flatten_tensors
from .matmul import matmul

PyTree = Any
# Scalar = Union[float, int, complex]


@struct.dataclass
class PyTreeArray:
    tree: PyTree
    treedefs: Any = struct.field(pytree_node=False)
    axes: Any = struct.field(pytree_node=False)  # Trees with the number of axes

    @property
    def T(self):
        return self.transpose()

    def _isnd(self, n):
        assert len(self.treedefs) == len(self.axes)
        return len(self.treedefs) == n

    def _is1d(self):
        return self._isnd(1)

    def _is2d(self):
        return self._isnd(2)

    @property
    def dtype(self):
        return jax.tree_map(jnp.dtype, self.tree)

    @property
    def _treedef(self):
        td = reduce(lambda s1, s2: s1.compose(s2), self.treedefs)
        assert td == jax.tree_structure(self.tree)
        return td

    def _leafdef(self, start, end=-1):
        td = _treedefs_compose(*self.treedefs[start:end])
        assert td == jax.tree_structure(self.tree)
        return td

    @property
    def treedef_l(self):
        # TODO deprecate?
        return self.treedefs[0]

    @property
    def treedef_r(self):
        # TODO deprecate?
        return self.treedefs[-1]

    @property
    def axes_l(self):
        # TODO deprecate?
        return self.axes[0]

    @property
    def axes_r(self):
        # TODO deprecate?
        return self.axes[-1]

    @property
    def ndim(self):
        n = len(self.treedefs)
        assert n == len(self.axes)
        return n

    def transpose(self):
        return transpose(self)

    # TODO @singledispatchmethod
    def __add__(self, t: PyTree):
        # elementwise or with a scalar
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x + t)
        elif isinstance(t, PyTreeArray):
            return self + t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.add, self.tree, t)
            return self.replace(tree=res)

    def __rmul__(self, t):
        return self * t

    def __radd__(self, t):
        return self + t

    def __rsub__(self, t):
        return (-self) + t

    def __neg__(self):
        return self._elementwise(lambda x: -x)

    def _elementwise(self, f):
        return self.replace(tree=jax.tree_map(f, self.tree))

    def __mul__(self, t: PyTree):
        # elementwise or with a scalar
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x * t)
        elif isinstance(t, jnp.ndarray) and t.ndim == 0 :
            return self._elementwise(lambda x: x * t)
        elif isinstance(t, PyTreeArray):
            # TODO check equal treedef_l and treedef_r, axes
            return self * t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.multiply, self.tree, t)
            return self.replace(tree=res)

    def __truediv__(self, t: PyTree):
        # elementwise or with a scalar
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x / t)
        elif isinstance(t, PyTreeArray):
            # TODO check equal treedef_l and treedef_r, axes
            return self / t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.divide, self.tree, t)
            return self.replace(tree=res)

    def __rtruediv__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: t / x)
        else:
            raise NotImplementedError

    def __sub__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x - t)
        elif isinstance(t, PyTreeArray):
            return self - t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.subtract, self.tree, t)
            return self.replace(tree=res)

    def __pow__(self, t):
        assert jnp.isscalar(t)
        return self._elementwise(lambda x: x ** t)

    def __getitem__(self, *args, **kwargs):
        return self.tree.__getitem__(*args, **kwargs)

    def __matmul__(self, pt2):
        if not isinstance(pt2, PyTreeArray):
            # assume its a pytree vector
            pt2 = PyTreeArray1(pt2)
        else:
            assert not pt2._isnd(0)
        return matmul(self, pt2)

    def conjugate(self):
        return self._elementwise(jnp.conj)

    def conj(self):
        return self.conjugate()

    @property
    def imag(self):
        return self._elementwise(jnp.imag)

    @property
    def real(self):
        return self._elementwise(jnp.real)

    # @property
    # def H(self):
    #     return self.T.conj()

    def _flatten_tensors(self):
        tree = amap(_flatten_tensors, self.tree, self.axes)
        _set1 = lambda x: jax.tree_map(lambda _: 1, x)
        axes = _set1(self.axes)
        return self.replace(tree=tree, axes=axes)

    def to_dense(self):
        return to_dense(self)

    def add_diag_scalar(self, a):
        assert self._is2d()
        assert self.treedef_l == self.treedef_r
        nl = self.treedef_l.num_leaves

        def _is_diag(i):
            return i % (nl + 1) == 0

        def _tree_map_diag(f, tree, is_diag):
            leaves, treedef = jax.tree_flatten(tree)
            return treedef.unflatten(f(l) if is_diag(i) else l for i, l in enumerate(leaves))

        def _add_diag_tensor(x, a):
            # TODO simpler ?
            s = x.shape
            n = x.ndim
            assert n % 2 == 0
            assert s[: n // 2] == s[n // 2 :]
            sl = s[: n // 2]
            _prod = lambda x: reduce(mul, x, 1)
            il = jnp.unravel_index(jnp.arange(_prod(sl)), sl)
            i = il + il
            return jax.ops.index_add(x, i, a)

        tree = _tree_map_diag(partial(_add_diag_tensor, a=a), self.tree, _is_diag)
        return self.replace(tree=tree)

    def sum(self, axis=0, keepdims=None):
        # for vectors only for now
        assert self.treedef_l == _arr_treedef
        tree = jax.tree_map(partial(jnp.sum, axis=axis, keepdims=keepdims), self.tree)
        if keepdims:
            n_ax = 0
        else:
            n_ax = 1 if isinstance(axis, int) else len(axis)
        axes_l = self.axes_l - n_ax
        return self.replace(tree=tree, axes=(axes_l,) + self.axes[1:])

    def astype(self, dtype_tree):
        if isinstance(dtype_tree, PyTreeArray):
            dtype_tree = dtype_tree.tree
        assert jax.tree_structure(dtype_tree) == self._treedef
        tree = jax.tree_multimap(lambda x, y: x.astype(y), self.tree, dtype_tree)
        return self.replace(tree=tree)

    # for the iterative solvers
    def __call__(self, vec):
        return self @ vec

    @property
    def at(self):
        from .atwrapper import _IndexUpdateHelper
        return _IndexUpdateHelper(self)


# for a (tree) vector
def PyTreeArray1(t):
    treedef_l = jax.tree_structure(t)
    # treedef_r = _arr_treedef
    axes_l = jax.tree_map(jnp.ndim, t)
    # axes_r = 0
    return PyTreeArray(t, (treedef_l,), (axes_l,))


# for a (normal) vector of (tree) vectors
def PyTreeArray2(t):
    treedef_l = _arr_treedef
    treedef_r = jax.tree_structure(t)
    axes_l = 1
    axes_r = jax.tree_map(lambda x: x - axes_l, jax.tree_map(jnp.ndim, t))
    return PyTreeArray(t, (treedef_l, treedef_r), (axes_l, axes_r))


# TODO eye_like / lazy add to diagonal
# TODO ignore flax FrozenDict in treedef comparison
# TODO ndim attr if treedefs are both * to emulate array
