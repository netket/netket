from netket.utils import jax_available


def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


if jax_available:
    from jax import tree_map as _tree_map

    def tree_map(fun, tree, *args, **kwargs):
        return _tree_map(lambda x: fun(x, *args, **kwargs), tree)


else:

    def tree_map(fun, tree, *args, **kwargs):
        """
        Maps all the leafs in the tree, applying the function with the leave as first
        positional argument.
        Any additional argument after the first two is forwarded to the function call.

        Args:
            fun: the function to apply to all leafs
            tree: the structure containing leafs. This can also be just a leaf
            *args: additional positional arguments passed to fun
            **kwargs: additional kw arguments passed to fun

        Returns:
            An equivalent tree, containing the result of the function call.
        """
        if tree is None:
            return None
        elif isinstance(tree, list):
            return [tree_map(fun, val, *args, **kwargs) for val in tree]
        elif isinstance(tree, tuple):
            if not hasattr(tree, "_fields"):
                # If it is not a namedtuple, recreate it as a tuple.
                return tuple(tree_map(fun, val, *args, **kwargs) for val in tree)
            else:
                # If it is a namedtuple, than keep that type information.
                return type(tree)(
                    *(tree_map(fun, val, *args, **kwargs) for val in tree)
                )
        elif isinstance(tree, dict):
            return {
                key: tree_map(fun, value, *args, **kwargs)
                for key, value in tree.items()
            }
        else:
            return fun(tree, *args, **kwargs)


if jax_available:
    from jax import tree_multimap

    def trees2_map(fun, tree1, tree2, *args, **kwargs):
        return tree_multimap(lambda x, y: fun(x, y, *args, **kwargs), tree1, tree2)


else:

    def trees2_map(fun, tree1, tree2, *args, **kwargs):
        """
        Maps all the leafs in the two trees, applying the function with the leafs of tree1
        as first argument and the leafs of tree2 as second argument
        Any additional argument after the first two is forwarded to the function call.

        This is usefull e.g. to sum the leafs of two trees

        Args:
            fun: the function to apply to all leafs
            tree1: the structure containing leafs. This can also be just a leaf
            tree2: the structure containing leafs. This can also be just a leaf
            *args: additional positional arguments passed to fun
            **kwargs: additional kw arguments passed to fun

        Returns:
            An equivalent tree, containing the result of the function call.
        """
        if tree1 is None:
            return None
        elif isinstance(tree1, list):
            return [
                trees2_map(fun, val1, val2, *args, **kwargs)
                for val1, val2 in zip(tree1, tree2)
            ]
        elif isinstance(tree1, tuple):
            return tuple(
                trees2_map(fun, val1, val2, *args, **kwargs)
                for val1, val2 in zip(tree1, tree2)
            )
        elif isinstance(tree1, dict):
            return {
                key: trees2_map(fun, val1, val2, *args, **kwargs)
                for (key, val1), (key2, val2) in zip(tree1.items(), tree2.items())
            }
        else:
            return fun(tree1, tree2, *args, **kwargs)


from netket.utils import jax_available

if jax_available:
    import jax
    from jax.tree_util import tree_flatten, tree_unflatten
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
