from netket.utils import wraps_legacy
from netket.legacy.optimizer import SR as SR_legacy
from netket.legacy.machine import AbstractMachine

from .sr_onthefly import SR_otf_cg, SR_otf_gmres

default_iterative = "cg"
# default_direct = "eigen"


@wraps_legacy(SR_legacy, "machine", AbstractMachine)
def SR(diag_shift=0.01, method=None, *, iterative=True, **kwargs):
    if method is None and iterative is True:
        method = default_iterative
    elif method is None and iterative is False:
        raise NotImplementedError(
            "Non-iterative methods for SR are no longer implemented"
        )

    if method == "cg":
        return SR_otf_cg(diag_shift, **kwargs)
    elif method == "gmres":
        return SR_otf_gmres(diag_shift, **kwargs)
    else:
        raise NotImplementedError("Only cg and gmres are implemented")
