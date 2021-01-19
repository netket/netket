import jax


def jit_if_singleproc(f, *args, **kwargs):
    if n_nodes == 1:
        return jax.jit(f, *args, **kwargs)
    else:
        return f


def get_afun_if_module(mod_or_fun, *args, **kwargs):
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun.apply
    else:
        return mod_or_fun
