import numpy as np

import jax
import jax.numpy as jnp

import netket

import pytest


def v1(x):
    return jnp.sum(jnp.exp(-(x**2)), axis=-1)


def v2(x):
    return jnp.sum(2.0 * jnp.exp(-(x**2)))


v2_vec = jax.vmap(v2)


hilb = netket.hilbert.Particle(N=1, L=jnp.inf, pbc=False)
hilb2 = netket.hilbert.Particle(N=2, L=5.0, pbc=True)

# potential operators
pot1 = netket.operator.PotentialEnergy(hilb, v1)
pot2 = netket.operator.PotentialEnergy(hilb, v2)
pot3 = netket.operator.PotentialEnergy(hilb2, v1)

# sum of potential operators
pottot = pot1 + pot2
pot10p52 = pot1 + 0.5 * pot2

# kinetic operators
kin1 = netket.operator.KineticEnergy(hilb, mass=20.0)
kin2 = netket.operator.KineticEnergy(hilb, mass=2.0)

# sum of kinetic operators
kintot = kin1 + kin2
kin10p52 = kin1 + 0.5 * kin2

# sum of potential and kinetic operators
etot = pottot + kintot
# TODO: make this recognize double appearing of same operator
etot2 = pot1 + pot2 + 2.0 * kin1 - kin1 + kin2

model1 = lambda p, x: 1.0
model2 = lambda p, x: jnp.sum(x**3)
model3 = lambda p, x: p * jnp.sum(x**3)
kinexact = lambda x: -0.5 * jnp.sum((3 * x**2) ** 2 + 6 * x, axis=-1)
kinexact2 = lambda p, x: -0.5 * jnp.sum((3 * p * x**2) ** 2 + 6 * p * x, axis=-1)


def test_equality():
    ops = [
        (
            netket.operator.PotentialEnergy(hilb, v1),
            netket.operator.PotentialEnergy(hilb, v1),
            netket.operator.PotentialEnergy(hilb, v2),
        ),
        (
            netket.operator.KineticEnergy(hilb, mass=20.0),
            netket.operator.KineticEnergy(hilb, mass=20.0),
            netket.operator.KineticEnergy(hilb, mass=2.0),
        ),
        (0.3 * pot1 + 0.2 * kin2, 0.3 * pot1 + 0.2 * kin2, 0.3 * pot1 + 0.2 * kin1),
    ]
    for op1a, op1b, op2 in ops:
        assert hash(op1a) == hash(op1b)
        assert hash(op1a) != hash(op2)
        assert op1a == op1b
        assert op1a != op2


def test_is_hermitean():
    epot = netket.operator.PotentialEnergy(hilb, v1)
    ekin = netket.operator.KineticEnergy(hilb, mass=20.0)
    etot = epot + ekin

    assert epot.is_hermitian
    assert ekin.is_hermitian
    assert etot.is_hermitian

    ekin = netket.operator.KineticEnergy(hilb, mass=20.0j)
    np.testing.assert_allclose(ekin.mass, 20.0j)
    assert not ekin.is_hermitian

    etot = epot + ekin
    assert not etot.is_hermitian


def test_potential_energy():
    x = jnp.zeros((1, 1))
    energy1 = pot1._expect_kernel(model1, None, x, pot1._pack_arguments())
    energy2 = pot2._expect_kernel(model1, None, x, pot2._pack_arguments())
    np.testing.assert_allclose(energy1, v1(x))
    np.testing.assert_allclose(energy2, v2_vec(x))
    with np.testing.assert_raises(NotImplementedError):
        pot1 + pot3


def test_kinetic_energy():
    x = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    energy1 = kin1._expect_kernel(model2, 0.0, x, kin1._pack_arguments())
    energy2 = kin1._expect_kernel(model3, 1.0 + 1.0j, x, kin1._pack_arguments())
    kinen1 = kinexact(x) / kin1.mass
    kinen2 = kinexact2(1.0 + 1.0j, x) / kin1.mass
    np.testing.assert_allclose(energy1, kinen1)
    np.testing.assert_allclose(energy2, kinen2)
    np.testing.assert_allclose(kin1.mass * kin1._pack_arguments(), 1.0)
    np.testing.assert_equal("KineticEnergy(m=20.0)", repr(kin1))


def test_sumoperator():
    x = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    potenergy = pottot._expect_kernel(model2, 0.0, x, pottot._pack_arguments())
    energy10p52 = pot10p52._expect_kernel(model2, 0.0, x, pot10p52._pack_arguments())

    np.testing.assert_allclose(potenergy, v1(x) + v2_vec(x))
    np.testing.assert_allclose(energy10p52, v1(x) + 0.5 * v2_vec(x))

    kinenergy = kintot._expect_kernel(model2, 0.0, x, kintot._pack_arguments())
    kinenergyex = kinexact(x) / kin1.mass + kinexact(x) / kin2.mass
    np.testing.assert_allclose(kinenergy, kinenergyex)

    kinen10p52 = kin10p52._expect_kernel(model2, 0.0, x, kin10p52._pack_arguments())
    kinenergy10p52ex = kinexact(x) / kin1.mass + 0.5 * kinexact(x) / kin2.mass
    np.testing.assert_allclose(kinen10p52, kinenergy10p52ex)

    enertot = etot._expect_kernel(model2, 0.0, x, etot._pack_arguments())
    enertot2 = etot2._expect_kernel(model2, 0.0, x, etot2._pack_arguments())
    enerexact = v1(x) + v2_vec(x) + kinexact(x) / kin1.mass + kinexact(x) / kin2.mass
    np.testing.assert_allclose(enertot, enerexact)
    np.testing.assert_allclose(enertot2, enerexact)

    with pytest.raises(AssertionError):
        netket.operator.SumOperator(etot, etot2, coefficients=[1.0])


@pytest.mark.parametrize(
    "dtype_x", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
@pytest.mark.parametrize(
    "dtype_op", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_dtype(dtype_op, dtype_x):
    dtype_energy = jnp.promote_types(dtype_op, dtype_x)
    x = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=dtype_x)

    pot = netket.operator.PotentialEnergy(hilb, v1, coefficient=1.0, dtype=dtype_op)
    assert pot.dtype == dtype_op
    assert pot.coefficient.dtype == dtype_op

    energy = pot1._expect_kernel(model3, 0.0, x, pot._pack_arguments())
    assert energy.dtype == dtype_energy

    kin = netket.operator.KineticEnergy(hilb, mass=1.0, dtype=dtype_op)
    assert kin.dtype == dtype_op
    assert kin.mass.dtype == dtype_op

    energy = kin._expect_kernel(model3, 0.0, x, kin._pack_arguments())
    assert energy.dtype == dtype_energy

    etot = pot + kin
    assert etot.dtype == dtype_op
    assert etot.coefficients.dtype == dtype_op

    energy = etot._expect_kernel(model3, 0.0, x, etot._pack_arguments())
    assert energy.dtype == dtype_energy
