import jax


class AbstractJaxKernel:
    def transition(self, key, states):
        batches = states.shape[0]
        keys = jax.random.split(key, batches)
        print("here.. states", states.shape)
        return jax.vmap(self.transition_singlechain, in_axes=(0, 0), out_axes=(0))(
            keys, states
        )


class _JaxLocalKernel(AbstractJaxKernel):
    def __init__(self, hilbert):
        self.hilbert = hilbert

    def transition_singlechain(self, key, state):
        print(key)
        print(state)
        print(state.shape)
        N = state.shape[0]

        keys = jax.random.split(key, 2)
        si = jax.random.randint(keys[0], shape=(1,), minval=0, maxval=N)
        new_state, _ = self.hilbert.jax_flip_state(keys[1], state, si)

        return new_state
