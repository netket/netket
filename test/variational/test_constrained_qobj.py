import numpy as np
import pytest
import jax

import netket as nk
from netket.hilbert.index.constrained_generic import compute_constrained_to_bare_conversion_table
from netket.hilbert.index.uniform_tensor import UniformTensorProductHilbertIndex

SEED = 2148364

@pytest.mark.skipif_mpi
def test_pure_state_qobj_constrained():
    pytest.importorskip("qutip")

    hi = nk.hilbert.Spin(s=0.5, N=2, total_sz=0)
    model = nk.models.RBM(alpha=1, param_dtype=float)
    vs = nk.vqs.FullSumState(hi, model)
    vs.init_parameters(seed=SEED)

    qobj = vs.to_qobj()
    assert qobj.type == "ket"
    assert qobj.dims[0] == list(hi.shape)
    assert qobj.dims[1] == [1 for _ in range(hi.size)]

    vec = qobj.data.toarray().reshape(-1)
    index = UniformTensorProductHilbertIndex(hi._local_states, hi.size)
    bare_numbers = np.asarray(
        compute_constrained_to_bare_conversion_table(index, hi.constraint)
    )
    mask = np.ones(vec.size, dtype=bool)
    mask[bare_numbers] = False
    assert np.allclose(vec[mask], 0)


@pytest.mark.skipif_mpi
def test_mixed_state_qobj_constrained():
    pytest.importorskip("qutip")

    hi = nk.hilbert.Spin(s=0.5, N=2, total_sz=0)
    model = nk.models.NDM(alpha=1, beta=1, param_dtype=float)
    sampler = nk.sampler.ExactSampler(nk.hilbert.DoubledHilbert(hi))
    vs = nk.vqs.MCMixedState(sampler, model, n_samples=10, seed=SEED)

    qobj = vs.to_qobj()
    assert qobj.type == "oper"
    assert qobj.dims[0] == list(hi.shape)
    assert qobj.dims[1] == list(hi.shape)

    mat = qobj.data.toarray()
    index = UniformTensorProductHilbertIndex(hi._local_states, hi.size)
    bare_numbers = np.asarray(
        compute_constrained_to_bare_conversion_table(index, hi.constraint)
    )
    full_size = int(np.prod(hi.shape))
    mask = np.ones((full_size, full_size), dtype=bool)
    mask[np.ix_(bare_numbers, bare_numbers)] = False
    assert np.allclose(mat[mask], 0)

