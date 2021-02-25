import netket.exact as exact
import netket.hilbert as hs
import netket.operator as op
import netket.graph as gr
import math as ma
import numpy as np


def build_rotation(hi, basis):
    localop = op.LocalOperator(hi, constant=1.0, dtype=complex)
    U_X = 1.0 / (ma.sqrt(2)) * np.asarray([[1.0, 1.0], [1.0, -1.0]])
    U_Y = 1.0 / (ma.sqrt(2)) * np.asarray([[1.0, -1j], [1.0, 1j]])

    N = hi.size

    assert len(basis) == hi.size
    for j in range(hi.size):
        if basis[j] == "X":
            localop *= op.LocalOperator(hi, U_X, [j])
        if basis[j] == "Y":
            localop *= op.LocalOperator(hi, U_Y, [j])
    return localop


def generate(N, n_basis=20, n_shots=1000, seed=1234):
    g = gr.Hypercube(length=N, n_dim=1, pbc=False)
    hi = hs.Spin(1 / 2, N=g.n_nodes)
    ha = op.Ising(hilbert=hi, h=1, graph=g)
    evals, evecs = exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)

    psi = evecs.reshape(-1)

    rotations = []
    training_samples = []
    training_bases = []

    np.random.seed(seed)

    for m in range(n_basis):
        basis = np.random.choice(
            list("XYZ"), size=N, p=[1.0 / N, 1.0 / N, (N - 2.0) / N]
        )

        rotation = build_rotation(hi, basis)
        psir = rotation.to_sparse().dot(psi)

        rand_n = np.random.choice(
            hi.n_states, p=np.square(np.absolute(psir)), size=n_shots
        )

        for rn in rand_n:
            training_samples.append(hi.numbers_to_states(rn))
        training_bases += [m] * n_shots

        rotations.append(rotation)

    return hi, tuple(rotations), training_samples, training_bases, ha, psi
