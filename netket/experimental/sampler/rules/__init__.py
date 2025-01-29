from netket.sampler.rules import (
    FermionHopRule as _deprecated_FermionHopRule,
)

_deprecations = {
    # June 2024, NetKet 3.13
    "ParticleExchangeRule": (
        "netket.experimental.sampler.rules.ParticleExchangeRule is deprecated: use "
        "netket.sampler.rules.FermionHopRule (netket >= 3.13)",
        _deprecated_FermionHopRule,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
