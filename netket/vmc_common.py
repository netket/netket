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
