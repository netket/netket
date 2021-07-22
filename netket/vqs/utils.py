import netket.jax as nkjax


def vjp_with_aux(fun, *args, mutable=False, **kwargs):
    if mutable is False:
        aux = None
        val, vjp_fun = nkjax.vjp(
            fun,
            *args,
            **kwargs,
        )
    else:
        val, vjp_fun, aux = nkjax.vjp(
            fun,
            *args,
            **kwargs,
            has_aux=True,
        )
    return val, vjp_fun, aux
