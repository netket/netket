import numpy as np
import netket.experimental as nkx
import netket.exact as exact
import netket.hilbert as hs
import netket.operator as op
import netket.graph as gr
from scipy.linalg import expm

from .. import common

pytestmark = common.skipif_distributed

SEED = 214748364


def _build_rotation(hi, basis, dtype=np.complex64):
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


def _thermal_state(ha, beta):
    rho = expm(-beta * ha.to_dense())
    rho = rho / np.trace(rho)
    return rho


def _generate_data(
    hi,
    rho,
    n_basis=20,
    n_shots=100,
):
    N = int(np.log2(rho.shape[0]))
    rotations = []
    training_samples = []
    training_bases = []

    np.random.seed(SEED)

    basis_list = []

    for m in range(n_basis):
        basis = np.random.choice(list("XYZ"), size=N, p=[1 / 3, 1 / 3, 1 / 3])

        basis_list.append(basis)

        rotation = _build_rotation(hi, basis)

        # rotate the state
        psir = np.sqrt(
            np.abs(np.diag(rotation.to_sparse() @ rho @ rotation.to_sparse().conj().T))
        ).flatten()

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

    hi = hs.DoubledHilbert(hi)

    return hi, rotations, training_samples, rho


def _setup_measurements(N, mode, n_basis=20):
    g = gr.Hypercube(length=N, n_dim=1, pbc=False)
    hi = hs.Spin(1 / 2, N=g.n_nodes)
    ha = op.Ising(hilbert=hi, h=1, graph=g)
    evals, evecs = exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    psi = evecs.reshape(-1, 1)
    if mode == "pure":
        rho = psi @ psi.conj().T
    elif mode == "mixed":
        rho = _thermal_state(ha, beta=1.0)
    else:
        raise ValueError("Invalid mode")
    hi, rotations, training_samples, rho = _generate_data(hi, rho, n_basis=n_basis)
    return hi, rotations, training_samples, rho


def test_raw_dataset():
    hi, rotations, training_samples, rho = _setup_measurements(3, "pure", n_basis=20)

    dataset = nkx.qsr.RawQuantumDataset((training_samples, rotations))
    assert len(dataset) == len(rotations)
    assert len(dataset.unique_bases()) == 20
    assert isinstance(repr(dataset), str)


def test_raw_dataset_preprocess():
    hi, rotations, training_samples, rho = _setup_measurements(3, "pure", n_basis=20)

    dataset = nkx.qsr.RawQuantumDataset((training_samples, rotations))
    dataset_processed = dataset.preprocess()
    assert len(dataset_processed) == dataset_processed.size
    assert len(dataset_processed) == len(dataset)

    assert isinstance(dataset_processed[1], type(dataset_processed))
    assert len(dataset_processed[1]) == 1
    assert len(dataset_processed[[1, 2]]) == 2
