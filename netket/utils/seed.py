import numpy as np


def random_seed() -> int:
    """
    generates a random seed
    """
    return np.random.randint(0, 1 << 32)


def random_PRNGKey():
    """
    generates a random jax PRNGKey
    """
    return jax.random.PRNGKey(random_seed())
