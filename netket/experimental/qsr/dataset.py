# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np

from numba import njit

from netket.operator import AbstractOperator, LocalOperator
from netket.hilbert import AbstractHilbert, Spin
from netket.utils import struct
from netket.utils.types import Array, DType

BaseType = Union[AbstractOperator, np.ndarray, str]


def _build_rotation(
    hi: Spin, basis: list | str, dtype: DType | None = np.complex64
) -> AbstractOperator:
    r"""
    Construct basis rotation operators from a Pauli string of "X", "Y", "Z" and "I".

    Args:
        hi: The Hilbert space
        basis: The Pauli string
        dtype: The data type of the returned operator

    Returns:
        The rotation operator
    """
    localop = LocalOperator(hi, constant=1.0, dtype=dtype)
    U_X = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, 1.0], [1.0, -1.0]])
    U_Y = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, -1j], [1.0, 1j]])

    assert len(basis) == hi.size

    for j, base in enumerate(basis):
        if base == "X":
            localop *= LocalOperator(hi, U_X, [j])
        elif base == "Y":
            localop *= LocalOperator(hi, U_Y, [j])
        elif base == "Z" or base == "I":
            pass

    return localop


def _canonicalize_bases_type(Us: list[BaseType] | np.ndarray) -> list[AbstractOperator]:
    r"""
    Check if the given bases are valid for the quantum state reconstruction driver.

    Args:
        Us (list or np.ndarray): A list of bases

    Raises:
        ValueError: When not given a list or np.ndarray
        TypeError: If the type of the operators is not a child of AbstractOperator
    """
    if not (isinstance(Us, list) or isinstance(Us, np.ndarray)):
        raise ValueError(
            "The bases should be a list or np.ndarray(dtype=object)" " of the bases."
        )

    if isinstance(Us[0], AbstractOperator):
        return Us

    if isinstance(Us[0], str):
        from netket.hilbert import Spin

        hilbert = Spin(0.5, N=len(Us[0]))
        N_samples = len(Us)

        _cache = {}
        _bases = np.empty(N_samples, dtype=object)

        for i, basis in enumerate(Us):
            if basis not in _cache:
                U = _build_rotation(hilbert, basis)
                _cache[basis] = U

            _bases[i] = _cache[basis]
        return _bases

    raise TypeError("Unknown type of measurement basis.")


def _convert_data(
    sigma_s: np.ndarray,
    Us: list[BaseType] | np.ndarray,
    mixed_state_target: bool | None = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Convert sampled states and rotation operators to a more direct computational format.
    Specifically, for each sampled state sigma_s, find all the states sigma_p that have non-zero
    matrix elements with sigma_s in the rotation operators Us. The corresponding
    non-zero matrix elements (mels) and the indices of sigma_p and Us that divide
    different sigma_s (secs) are also returned.

    For pure states, we directly return sigma_p, mels and secs, so that one can use them
    to compute <sigma_s|U|sigma_p> and sum over sigma_p. For mixed states, we return sigma_p x sigma_p,
    mels and mels.conj() and the corresponding secs, so that one can use them to compute
    <sigma_s|U|sigma_p><sigma_p'|U|sigma_s> and sum over both sigma_p and sigma_p'.

    Args:
        sigma_s (np.ndarray): The states
        Us (np.ndarray or list): The list of rotations
        mixed_state_target (bool): Whether to use mixed states or not

    Returns:
        sigma_p (np.ndarray): All the states that have non-zero matrix elements
        with the input states sigma_s in the rotation operators Us.
        mels (np.ndarray): The corresponding non-zero matrix elements.
        secs (np.ndarray): Indices of sigma_p and Us that divide different sigma_s.
        MAX_LEN (int): The maximum number of connected states.
    """
    Us = _canonicalize_bases_type(Us)

    # Error message when user tries to convert less or more sigmas than Us
    assert (
        sigma_s.shape[0] == Us.shape[0]
    ), "The number of samples should be equal to the number of rotations."

    N = sigma_s.shape[-1]
    sigma_s = sigma_s.reshape(-1, N)
    Nb = sigma_s.shape[0]
    N_target = N + N * mixed_state_target  # N or 2N if mixed state

    # constant number of connected states per operator
    Nc = Us[0].hilbert.local_size
    sigma_p = np.zeros((0, N_target), dtype=sigma_s.dtype)
    mels = np.zeros((0,), dtype=Us[0].dtype)
    secs = np.zeros(Nb + 1, dtype=np.intp)
    MAX_LEN = 0

    last_i = 0
    for i, (sigma, U) in enumerate(zip(sigma_s, Us)):
        sigma_p_i, mels_i = U.get_conn(sigma)

        if not mixed_state_target:
            Nc = mels_i.size
        else:
            # size of the cartesian product sigma_p x sigma_p
            Nc = mels_i.size**2

        sigma_p = np.resize(sigma_p, (last_i + Nc, N_target))
        mels = np.resize(mels, (last_i + Nc,))

        if not mixed_state_target:
            sigma_p[last_i:, :] = sigma_p_i
            # <sigma_s|U|sigma_p>
            mels[last_i:] = mels_i
        else:
            # indices of the cartesian product
            x, y = np.meshgrid(np.arange(mels_i.size), np.arange(mels_i.size))
            sigma_p[last_i:, :] = np.hstack(
                [sigma_p_i[x.flatten()], sigma_p_i[y.flatten()]]
            )
            # <sigma_s|U|sigma_p><sigma_p'|U|sigma_s>
            mels[last_i:] = np.prod(
                np.stack(
                    [mels_i[x.flatten()], np.conjugate(mels_i[y.flatten()])], axis=-1
                ),
                axis=-1,
            )

        secs[i] = last_i
        last_i = last_i + Nc
        MAX_LEN = max(Nc, MAX_LEN)

    sigma_p = np.resize(sigma_p, (last_i + MAX_LEN, N_target))
    mels = np.resize(mels, (last_i + MAX_LEN,))
    sigma_p[last_i + Nc :, :] = 0.0
    mels[last_i + Nc :] = 0.0
    secs[-1] = last_i  # + MAX_LEN

    return sigma_p, mels, secs, MAX_LEN


@njit
def _compose_sampled_data(
    sigma_p: np.ndarray,
    mels: np.ndarray,
    secs: np.ndarray,
    MAX_LEN: int,
    sampled_indices: np.ndarray,
    min_padding_factor: int | None = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Given the sampled indices, select the corresponding data from sigma_p, mels and secs.

    Args:
        sigma_p (np.ndarray): All the states that have non-zero matrix elements
        with the input states sigma_s in the rotation operators Us.
        mels (np.ndarray): The corresponding non-zero matrix elements.
        secs (np.ndarray): Indices of sigma_p and Us that divide different sigma_s.
        MAX_LEN (int): The maximum number of connected states.
        sampled_indices (np.ndarray): The indices of the sampled states.
        min_padding_factor (int): The minimum padding factor.

    Returns:
        The sampled sigma_p, mels, secs and MAX_LEN.
    """
    N_samples = sampled_indices.size
    N = sigma_p.shape[-1]

    _sigma_p = np.zeros((N_samples * MAX_LEN, N), dtype=sigma_p.dtype)
    _mels = np.zeros((N_samples * MAX_LEN,), dtype=mels.dtype)
    _secs = np.zeros((N_samples + 1,), dtype=secs.dtype)
    _maxlen = 0

    last_i = 0
    for n, i in enumerate(sampled_indices):
        start_i, end_i = secs[i], secs[i + 1]
        len_i = end_i - start_i

        # print(f"{n}, {i}, {last_i}, {start_i}, {end_i}")
        _sigma_p[last_i : last_i + len_i, :] = sigma_p[start_i:end_i, :]
        _mels[last_i : last_i + len_i] = mels[start_i:end_i]

        last_i = last_i + len_i
        _secs[n + 1] = last_i

        _maxlen = max(_maxlen, len_i)

    padding_factor = max(
        MAX_LEN, min_padding_factor
    )  # minimum padding at 64 to avoid excessive recompilation
    padded_size = padding_factor * int(np.ceil(last_i / padding_factor))

    _sigma_p = _sigma_p[:padded_size, :]
    _mels = _mels[:padded_size]

    return _sigma_p, _mels, _secs, _maxlen


class RawQuantumDataset:
    """
    Class used to store a dataset of Quantum shots, usually taken from a quantum computer
    or simulator.
    """

    def __init__(self, dataset: tuple[list, list]):
        if not isinstance(dataset, tuple) or len(dataset) != 2:
            raise TypeError("not a tuple of length 2")

        measurements, bases = dataset
        bases = _canonicalize_bases_type(bases)

        if measurements.ndim != 2:
            raise ValueError(
                "Measurements should be an array with 2 dimensions, where"
                "(measurement_i, N_qubits)."
            )

        # Error message when user tries to convert less or more sigmas than Us
        if measurements.shape[0] != len(bases):
            raise ValueError(
                f"The number of measurements ({measurements.shape[0]}) "
                f"should be equal to the number of rotations ({len(bases)})."
            )

        self._measurements = measurements
        self._bases = bases

    @property
    def bases(self):
        """
        Returns a 1D numpy array of the bases used to measure the respective measurement
        returned by the `{self.measurements}` property.
        """
        return self._bases

    @property
    def measurements(self):
        """
        Returns a 2D numpy array containing the measurement outcome in the respective basis
        given by `{self.bases}` property.
        """
        return self._measurements

    def __len__(self):
        return len(self.bases)

    def unique_bases(self):
        """
        Returns the list of unique bases present in {ref}`self.bases`.
        """
        unique_bases = []
        _last_basis = None
        for b in self.bases:
            if b == _last_basis:
                continue
            elif b in unique_bases:
                continue
            else:
                unique_bases.append(b)
        return np.array(unique_bases)

    def preprocess(
        self, *, mixed_state_target: bool = False, hilbert: AbstractHilbert = None
    ):
        """
        Constructs the `ProcessedQuantumDataset` object with the entirety of this dataset.

        The `ProcessedQuantumDataset` holds the measurements and operators in a format that
        can be more efficiently used to compute and optimise the KL.
        """
        sigma_p, mels, secs, MAX_LEN = _convert_data(
            self.measurements, self.bases, mixed_state_target
        )

        if hilbert is None:
            hilbert = Spin(0.5, sigma_p.shape[-1])

        return ProcessedQuantumDataset(
            hilbert, sigma_p, mels, secs, MAX_LEN, mixed_state_target
        )

    def __repr__(self):
        return f"RawQuantumDataset(N_measurements={len(self)})"


@struct.dataclass
class ProcessedQuantumDataset:
    hilbert: AbstractHilbert
    """
    The global computational basis of those measurements
    """

    sigma_p: Array
    """
    The precomputed connected elements of the rotations for the measured bitstrings
    """

    mels: Array
    """
    The precomputed matrix elements of the rotations for the measured bitstrings
    """

    secs: Array
    """The secs"""

    max_len: int = struct.field(pytree_node=False)

    # training_samples_n : int = struct.field(pytree_node=False)

    mixed_state_target: bool = struct.field(pytree_node=False)

    @property
    def size(self) -> int:
        return len(self.secs) - 1

    def subsample(self, batch_size, *, rng, batch_sample_replace: bool = True):
        # sample training data for pos grad
        sampled_indices = np.sort(
            rng.choice(
                self.size,
                size=(batch_size,),
                replace=batch_sample_replace,
            )
        )

        return self[sampled_indices]

    def __len__(self):
        return len(self.secs) - 1

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = np.array([idx])
        elif isinstance(idx, list):
            idx = np.array(idx)
        elif not isinstance(idx, np.ndarray):
            raise TypeError(
                "fThe accessor only works with scalars and 1D-arrays, but it was a `{type(idx)}`."
            )

        if idx.ndim != 1:
            raise TypeError(
                f"The indices must be a 1D array, but it was `idx.shape={idx.shape}`."
            )

        sigma_p, mels, secs, maxlen = _compose_sampled_data(
            self.sigma_p,
            self.mels,
            self.secs,
            self.max_len,
            idx,
        )

        return ProcessedQuantumDataset(
            self.hilbert, sigma_p, mels, secs, maxlen, self.mixed_state_target
        )
