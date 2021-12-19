from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import unfreeze


# TODO(19-dec-2021): Deprecate: eventually remove this
def update_GCNN_parity(params):
    """Adds biases of parity-flip layers to the corresponding no-flip layers.
    Corrects for changes in GCNN_parity due to PR #1030 in NetKet 3.3.

    Args:
        params: a parameter pytree
    """
    # unfreeze just in case, doesn't break with a plain dict
    params = flatten_dict(unfreeze(params))
    to_remove = []
    for path in params:
        if (
            len(path) > 1
            and path[-2].startswith("equivariant_layers_flip")
            and path[-1] == "bias"
        ):
            alt_path = (
                *path[:-2],
                path[-2].replace("equivariant_layers_flip", "equivariant_layers"),
                path[-1],
            )
            params[alt_path] = params[alt_path] + params[path]
            to_remove.append(path)
    for path in to_remove:
        del params[path]
    return unflatten_dict(params)
