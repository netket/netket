import jax


class _JaxLocalKernel:
    def __init__(self, local_states, size):
        self.local_states = jax.numpy.sort(jax.numpy.array(local_states))
        self.size = size
        self.n_states = self.local_states.size

    def transition(self, key, state):

        keys = jax.random.split(key, 2)
        si = jax.random.randint(keys[0], shape=(1,), minval=0, maxval=self.size)
        rs = jax.random.randint(keys[1], shape=(1,), minval=0, maxval=self.n_states - 1)

        return jax.ops.index_update(
            state, si, self.local_states[rs + (self.local_states[rs] >= state[si])]
        )

    def random_state(self, key, state):
        keys = jax.random.split(key, 2)

        rs = jax.random.randint(
            keys[1], shape=(self.size,), minval=0, maxval=self.n_states
        )

        return keys[0], self.local_states[rs]
