import netket.exact as exact
import netket.hilbert as hs
import netket.operator as op
import netket.graph as gr
import numpy as np


def build_rotation(hi, basis, dtype=np.complex64):
    localop = op.LocalOperator(hi, constant=1.0, dtype=dtype)
    U_X = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, 1.0], [1.0, -1.0]])
    U_Y = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, -1j], [1.0, 1j]])

    assert len(basis) == hi.size
    for j in range(hi.size):
        if basis[j] == "X":
            localop *= op.LocalOperator(hi, U_X, [j])
        if basis[j] == "Y":
            localop *= op.LocalOperator(hi, U_Y, [j])
    return localop


def generate(
    N,
    n_basis=20,
    n_shots=1000,
    seed=None,
    basis_generator=None,
    psi=None,
    rho=None,
    return_basis=False,
):
    g = gr.Hypercube(length=N, n_dim=1, pbc=False)
    hi = hs.Spin(1 / 2, N=g.n_nodes)
    ha = None

    if psi is None and rho is None:
        # if not specify any target state
        # build a 1d open Ising chain
        ha = op.Ising(hilbert=hi, h=1, graph=g)
        evals, evecs = exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)

        psi = evecs.reshape(-1)

    rotations = []
    training_samples = []
    training_bases = []

    if seed is not None:
        np.random.seed(seed)

    basis_list = []

    for m in range(n_basis):
        if basis_generator is None:
            # if not specify any measurement strategy
            basis = np.random.choice(
                list("XYZ"), size=N, p=[1.0 / N, 1.0 / N, (N - 2.0) / N]
            )
        else:
            # basis generator should generate a basis per call
            basis = basis_generator()

        basis_list.append(basis)

        rotation = build_rotation(hi, basis)

        # rotate the state
        if rho is not None:
            psir = np.sqrt(
                np.abs(
                    np.diag(rotation.to_sparse() @ rho @ rotation.to_sparse().conj().T)
                )
            ).flatten()
        else:
            psir = rotation.to_sparse().dot(psi)

        # compute the Born probabilities
        p = np.square(np.absolute(psir))
        p /= np.sum(p)

        # sampling
        rand_n = np.random.choice(hi.n_states, p=p, size=n_shots)

        for rn in rand_n:
            training_samples.append(hi.numbers_to_states(rn))
        training_bases += [m] * n_shots

        rotations.append(rotation)

    rotations = np.array(rotations)
    training_bases = np.array(training_bases)
    training_samples = np.array(training_samples).astype(np.float32)
    basis_list = np.array(basis_list)

    rotations = rotations[training_bases]

    basis_list = basis_list[training_bases]

    if rho is not None:
        # use doubled hilbert space for mixed states
        hi = hs.DoubledHilbert(hi)
        psi = rho

    psi = psi

    if return_basis:
        # also return the basis list
        return hi, rotations, training_samples, ha, psi, basis_list

    return hi, rotations, training_samples, ha, psi
