__all__ = [
    "AbstractVariationalDriver",
    "AbstractNGDDriver",
    "VMC",
    "InfidelityOptimizer",
    "VMC_NG",
    "InfidelityOptimizerNG",
    "InfidelityOptimizerNG_FS",
]

from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver as AbstractVariationalDriver,
)
from advanced_drivers._src.driver.vmc import (
    VMC as VMC,
)
from advanced_drivers._src.driver.infidelity import (
    InfidelityOptimizer as InfidelityOptimizer,
)

from advanced_drivers._src.driver.ngd.driver_abstract_ngd import (
    AbstractNGDDriver as AbstractNGDDriver,
)
from advanced_drivers._src.driver.ngd.driver_infidelity_ngd import (
    InfidelityOptimizerNG as InfidelityOptimizerNG,
)
from advanced_drivers._src.driver.ngd.driver_vmc_ngd import (
    VMC_NG as VMC_NG,
)
from advanced_drivers._src.driver.ngd.driver_infidelity_ngd_fullsum import (
    InfidelityOptimizerNG_FS as InfidelityOptimizerNG_FS,
)


from advanced_drivers._src.distribution.default import (
    DefaultDistribution as _deprecated_default_distribution,
)

from advanced_drivers._src.distribution.overdispersed import (
    OverdispersedDistribution as _deprecated_overdispersed_distribution,
    OverdispersedMixtureDistribution as _deprecated_overdispersed_mixture_distribution,
)
from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

_deprecations = {
    # April 2025, only Luca uses it. quickly remove it
    "InfidelityFullSum": (
        "advanced_drivers.driver.InfidelityFullSum is deprecated: use "
        "advanced_drivers.driver.InfidelityOptimizerNG_FS instead",
        InfidelityOptimizerNG_FS,
    ),
    # June 2025
    "default_distribution": (
        "advd.driver.default_distribution is deprecated: use "
        "advd.distribution.DefaultDistribution",
        _deprecated_default_distribution,
    ),
    "overdispersed_distribution": (
        "advd.driver.OverdispersedDistribution is deprecated: use "
        "advd.distribution.overdispersed_distribution",
        _deprecated_overdispersed_distribution,
    ),
    "overdispersed_mixture_distribution": (
        "advd.driver.overdispersed_mixture_distribution is deprecated: use "
        "advd.distribution.OverdispersedMixtureDistribution",
        _deprecated_overdispersed_mixture_distribution,
    ),
}


__getattr__ = _deprecation_getattr(__name__, _deprecations)

del _deprecation_getattr
