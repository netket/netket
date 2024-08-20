import pytest

import numpy as np
import netket as nk
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


def _setup_measurements(N, mode):
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
    hi, rotations, training_samples, rho = _generate_data(hi, rho)
    return hi, rotations, training_samples, rho


def _setup_driver(N, mode, control_variate_update_freq=10, chunk_size=97):
    hi, rotations, training_samples, rho = _setup_measurements(N, mode)

    if mode == "pure":
        ma = nk.models.RBM(alpha=1, param_dtype=complex)
        sa = nk.sampler.MetropolisLocal(hilbert=hi.physical)
        vs = nk.vqs.MCState(sa, ma, n_samples=1000, seed=SEED)
    elif mode == "mixed":
        ma = nk.models.NDM(alpha=1, param_dtype=complex)
        sa = nk.sampler.MetropolisLocal(hilbert=hi)
        vs = nk.vqs.MCMixedState(sa, ma, n_samples=1000, seed=SEED)
    else:
        raise ValueError("Invalid mode")

    op = nk.optimizer.Adam(learning_rate=0.01)

    driver = nkx.QSR(
        (training_samples, rotations),
        training_batch_size=100,
        optimizer=op,
        variational_state=vs,
        control_variate_update_freq=control_variate_update_freq,
        chunk_size=chunk_size,
    )
    return driver, rho


####


@pytest.mark.parametrize("chunk_size", [None, 30, 70, 110])
def test_pure_qsr(chunk_size):
    N = 3
    driver, rho = _setup_driver(N, "pure", chunk_size=chunk_size)
    assert driver.mixed_states is False
    driver.run(n_iter=20, out="test_pure_qsr.out")


def test_mixed_qsr():
    N = 3
    driver, rho = _setup_driver(N, "mixed")
    assert driver.mixed_states is True
    driver.run(n_iter=20, out="test_pure_qsr.out")


def test_pure_KL():
    N = 3
    driver, rho = _setup_driver(N, "pure")
    driver.run(n_iter=20, out="test_pure_qsr.out")
    driver.KL(rho, n_shots=100)
    driver.KL_whole_training_set(rho, n_shots=100)
    driver.KL_exact(rho, n_shots=100)


def test_mixed_KL():
    N = 3
    driver, rho = _setup_driver(N, "mixed")
    driver.run(n_iter=20, out="test_pure_qsr.out")
    driver.KL(rho, n_shots=100)
    driver.KL_whole_training_set(rho, n_shots=100)
    driver.KL_exact(rho, n_shots=100)
