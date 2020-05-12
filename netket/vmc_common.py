def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


def tree_map(fun, tree):
    if tree is None:
        return None
    elif isinstance(tree, list):
        return [tree_map(fun, val) for val in tree]
    elif isinstance(tree, tuple):
        return tuple(tree_map(fun, val) for val in tree)
    elif isinstance(tree, dict):
        return {key: tree_map(fun, value) for key, value in tree.items()}
    else:
        return fun(tree)


from netket.utils import jax_available

if jax_available:
    import jax
    from jax.tree_util import tree_flatten, tree_unflatten, tree_map
    import jax.numpy as jnp

    @jax.jit
    def jax_shape_for_update(update, shape_like):
        r"""Reshapes grads from array to tree like structure if neccesary for update

        Args:
            grads: a 1d jax/numpy array
            shape_like: this as in instance having the same type and shape of
                        the desired conversion.

        Returns:
            A possibly non-flat structure of jax arrays containing a copy of data
            compatible with the given shape if jax_available and a copy of update otherwise
        """

        shf, tree = tree_flatten(shape_like)

        updatelist = []
        k = 0
        for s in shf:
            size = s.size
            updatelist.append(jnp.asarray(update[k : k + size]).reshape(s.shape))
            k += size

        return tree_unflatten(tree, updatelist)
