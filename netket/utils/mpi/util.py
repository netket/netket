import jax


def mpi_tree_map(mpi_fun, x, *, token=None):
    """A version of tree map that also takes a token
    and enforces a specific execution order by weaving the
    token through every mpi_fun.

    Args:
        mpi_fun: The mpi reduction function that must take a
            token kw argument.
        x: The input PyTree
        token: The optional initial token

    Result:
        the output PyTree and the output token.
    """
    token_container = [token]

    def fun(y):
        res, new_token = mpi_fun(y, token=token_container[0])
        token_container[0] = new_token
        return res

    res = jax.tree_map(fun, x)
    return res, token_container[0]
