from .diagonal import Diagonal
from .abstract_density_matrix import AbstractDensityMatrix
from .rbm import RbmSpin

from netket.utils import jax_available


from .jax import Jax, NdmSpin, NdmSpinPhase, JaxRbmSpin
from .flax import Flax
