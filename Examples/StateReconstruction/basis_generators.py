import numpy as np
from itertools import product


# example basis generators
class BasisGenerator:
    def __init__(self, N):
        self.basis = []
        self.ind = 0
        self.N = N

    def __call__(self):
        self.ind = (self.ind + 1) % len(self.basis)
        return self.basis[self.ind - 1]

    def __len__(self):
        return len(self.basis)


class BasisGeneratorFull(BasisGenerator):
    def __init__(self, N):
        super().__init__(N)
        self.basis = list(product("XYZ", repeat=N))


class BasisGeneratorBound(BasisGenerator):
    def __init__(self, N, bound, score_dict={"X": 1, "Y": 1, "Z": 0}):
        super().__init__(N)
        self.basis = np.array(list(product("XYZ", repeat=N)))
        self.score = np.array(
            list(product((score_dict["X"], score_dict["Y"], score_dict["Z"]), repeat=N))
        )
        self.basis = self.basis[np.sum(self.score, axis=-1) <= bound].tolist()


class BasisGeneratorRandom(BasisGenerator):
    def __init__(self, N, n_basis, probs=[1 / 3, 1 / 3, 1 / 3]):
        super().__init__(N)
        self.basis = [
            np.random.choice(list("XYZ"), size=N, p=probs) for _ in range(n_basis)
        ]
