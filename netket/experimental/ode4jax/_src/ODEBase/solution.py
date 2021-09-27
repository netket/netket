import jax
import jax.numpy as jnp

from netket.utils import struct

from netket.utils.types import Array, PyTree

@struct.dataclass(_frozen=False)
class ODESolution:
	u : list
	"""
	History PyTree of the objects that are stored. The leading dimension is the dimension
	of the time
	"""
	t : list
	"""
	Time at which those objects are stored
	"""

	last_id : int = 0
	"""
	Last id stored
	"""


	def set(self, iter_count, t, val):
		self.t = self.t.at[iter_count].set(t)
		self.u = self.u.at[iter_count].set(val)
		self.last_id = iter_count

	@staticmethod
	def make(u, n):

		def prealloc_arr(x):
			return jnp.zeros((n,)+x.shape, dtype=x.dtype)
		u_prealloc = jax.tree_map(prealloc_arr, u)

		t_prealloc = jnp.zeros((n,), dtype=float)

		return ODESolution(u_prealloc, t_prealloc, last_id=0)
