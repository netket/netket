# Copyright 2022 The NetKet Authors - All rights reserved.
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

from netket.operator import LocalOperator
import math
import numpy as np
import time

_chebysheb18_constants = {
    "a11": 3 / 25,
    "a21": -0.00877476096879703859j,
    "a31": -0.00097848453523780954,
    "b11": -0.66040840760771318751j,
    "b21": -1.09302278471564897987,
    "b31": 0.25377155817710873323j,
    "b61": 0.00054374267434731225,
    "b02": -2.58175430371188142440,
    "b12": -1.73033278310812419209j,
    "b22": -0.07673476833423340755,
    "b32": -0.00261502969893897079j,
    "b62": -0.00003400011993049304,
    "b03": 2.92377758396553673559,
    "b13": 1.44513300347488268510j,
    "b23": 0.12408183566550450221,
    "b33": -0.01957157093642723948j,
    "b63": 0.00002425253007433925,
    "b24": -0.123953695858283131480j,
    "b34": -0.011202694841085592373,
    "b64": -0.000012367240538259896j,
}


def _one_norm(op: LocalOperator) -> float:
    """
    Computes the 1-norm of the matrix representation of the LocalOperator

    Parameters
    ----------
    op : LocalOperator
        A local operator.

    Returns
    -------
    float
        1-norm of the local operator.
    """
    hilbert = op.hilbert
    operators_dict = op._operators_dict
    local_sizes = np.asarray(hilbert.shape)
    one_norm = 0
    # TODO: this can be hugely accelerated
    for i in range(math.prod(local_sizes)):
        ket = np.asarray(np.unravel_index(i, local_sizes))
        value = 0
        for acting_on, operator in operators_dict.items():
            acting_on = np.atleast_1d(acting_on)
            col_idx = np.ravel_multi_index(
                tuple(ket[acting_on]),
                tuple(local_sizes[acting_on]),
            )
            value += np.abs(np.sum(operator[:, col_idx]))
        if value > one_norm:
            one_norm = value
    return one_norm


def _chebysheb18(A: LocalOperator) -> LocalOperator:
    """
    Computes the Chebyshev polynomial of degree 18 according to Bader, P., Blanes, S., Casas, F., & Seydaoğlu, M. (2021).
    An efficient algorithm to compute the exponential of skew-Hermitian matrices for the time integration of the
    Schrödinger equation. Mathematics and Computers in Simulation, 194, 383-400. https://doi.org/10.48550/arxiv.2103.10132

    Parameters
    ----------
    A : LocalOperator
        A local operator

    Returns
    -------
    LocalOperator
        Chebysheb polynomial of A with machine precision when the spectrum of A is bounded by 2.212
    """
    t0 = time.time()
    A2 = A * A
    t1 = time.time()
    print("A2 computed in", t1 - t0)
    A3 = A2 * A
    t2 = time.time()
    print("A3 computed in", t2 - t1)
    A6 = A3 * A3
    t3 = time.time()
    print("A6 computed in", t3 - t2)
    B1 = (
        _chebysheb18_constants["a11"] * A
        + _chebysheb18_constants["a21"] * A2
        + _chebysheb18_constants["a31"] * A3
    )
    t4 = time.time()
    print("B1 computed in", t4 - t3)
    B2 = (
        _chebysheb18_constants["b11"] * A
        + _chebysheb18_constants["b21"] * A2
        + _chebysheb18_constants["b31"] * A3
        + _chebysheb18_constants["b61"] * A6
    )
    t5 = time.time()
    print("B2 computed in", t5 - t4)
    B3 = (
        _chebysheb18_constants["b02"]
        + _chebysheb18_constants["b12"] * A
        + _chebysheb18_constants["b22"] * A2
        + _chebysheb18_constants["b32"] * A3
        + _chebysheb18_constants["b62"] * A6
    )
    t6 = time.time()
    print("B3 computed in", t6 - t5)
    B4 = (
        _chebysheb18_constants["b03"]
        + _chebysheb18_constants["b13"] * A
        + _chebysheb18_constants["b23"] * A2
        + _chebysheb18_constants["b33"] * A3
        + _chebysheb18_constants["b63"] * A6
    )
    t7 = time.time()
    print("B4 computed in", t7 - t6)
    B5 = (
        _chebysheb18_constants["b24"] * A2
        + _chebysheb18_constants["b34"] * A3
        + _chebysheb18_constants["b64"] * A6
    )
    t8 = time.time()
    print("B5 computed in", t8 - t7)
    A9 = B1 * B5 + B4
    t9 = time.time()
    print("A9 computed in", t9 - t8)
    res = B2 + (B3 + A9) * A9
    t10 = time.time()
    print("Result computed in", t10 - t9)
    return res


def propagator(op: LocalOperator, t: float) -> LocalOperator:
    """
    Computes the time-evolution propagator U(t) = exp(-i*op*t)

    Parameters
    ----------
    op : LocalOperator
        A local operator, typically a Hamiltonian.
    t : float
        Total evolution time.

    Returns
    -------
    LocalOperator
        Propagator.
    """
    op = op * t
    β = _one_norm(op)
    print("One norm of operator is", β)
    s = 0
    while β > 2.212:
        print("Halving beta")
        s += 1
        β = 0.5 * β
    Ut = _chebysheb18(op / (2**s))
    for i in range(s):
        Ut = Ut * Ut
    return Ut


if __name__ == "__main__":
    import netket as nk
    import time
    import matplotlib.pyplot as plt

    # Reproduce article figure
    x = np.linspace(-2.212, 2.212, 200)
    y = [np.abs(np.exp(-1j * xp) - _chebysheb18(xp)) for xp in x]
    plt.plot(x, y)
    plt.show()

    hi = nk.hilbert.Fock(N=4, n_max=3)
    op = nk.operator.BoseHubbard(hi, nk.graph.Chain(4), U=1)
    oplocal = op.to_local_operator()
    ft = time.time()
    print("One norm of dense", np.linalg.norm(op.to_dense(), ord=1))
    st = time.time()
    print("Dense took", st - ft)
    print("One norm by counting", _one_norm(oplocal))
    print("Counting took", time.time() - st)
    dt = 0.1

    U = propagator(oplocal, dt)
    H = op.to_qobj()
    Uqutip = (-1j * dt * H).expm().full()
    Unk = U.to_dense()
    print(
        "Frobenius norm of difference between direct exponentiation and approximated exponentiation",
        np.linalg.norm(Uqutip - Unk, ord=2),
    )
