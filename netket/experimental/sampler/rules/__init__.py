from netket.sampler.rules import (
    ParticleExchangeRule as _deprecated_ParticleExchangeRule,
)

_deprecations = {
    # June 2024, NetKet 3.13
    "ParticleExchangeRule": (
        "netket.experimental.sampler.rules.ParticleExchangeRule is deprecated: use "
        "netket.sampler.rules.ParticleExchangeRule (netket >= 3.13)",
        _deprecated_ParticleExchangeRule,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
