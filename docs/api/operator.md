(netket_operator_api)=
# netket.operator

```{eval-rst}
.. currentmodule:: netket.operator

```

The Operator module defines the common interfaces used to interact with quantum operators and super-operators, as well as several concrete implementations of different operators such as netket.hilbert.LocalOperator, netket.hilbert.Ising and others.

NetKet’s operators are all sub-classes of the abstract class netket.hilbert.AbstractOperator, which defines a small set of API respected by all implementations. The inheritance diagram for the class hierarchy of the Operators included with NetKet is shown below (you can click on the nodes in the graph to go to their API documentation page). Dashed nodes represent abstract classes that cannot be instantiated, while the others are concrete and they can be instantiated.



```{eval-rst}
.. inheritance-diagram:: netket.operator netket.experimental.operator
   :top-classes: netket.operator.AbstractOperator
   :parts: 1

```

## Abstract Classes

Below you find a list of all public classes defined in this module
Those classes cannot be directly instantiated, but you can inherit from one of them if you want to define new hilbert spaces.

```{eval-rst}
.. currentmodule:: netket.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   AbstractOperator
   AbstractSuperOperator
   DiscreteOperator
   DiscreteJaxOperator
   ContinuousOperator
```

## Concrete Classes

Below you find a list of all concrete Operators that you can create on {class}`~netket.hilbert.DiscreteHilbert` spaces.

```{eval-rst}
.. currentmodule:: netket.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   BoseHubbard
   BoseHubbardNumba
   GraphOperator
   LocalOperator
   LocalOperatorNumba
   Ising
   IsingNumba
   Heisenberg
   PauliStrings
   PauliStringsNumba
   LocalLiouvillian

```

### Fermions

Operators and functions to work with fermions are the following:

```{eval-rst}
.. autosummary::
   :toctree: _generated/operator
   :template: class
   :nosignatures:

   FermionOperator2nd
   FermionOperator2ndNumba
   fermion.create
   fermion.destroy
   fermion.number
```

In the experimental submodule there is also an implementation of a particle-number conserving operator which can be more efficient than the generic  {class}`~netket.experimental.operator.FermionOperator2ndJax`.

```{eval-rst}
.. currentmodule:: netket

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   experimental.operator.ParticleNumberConservingFermioperator2nd
   experimental.operator.ParticleNumberAndSpinConservingFermioperator2nd
   experimental.operator.FermiHubbardJax
```


Note in particular the pyscf module that can be used to convert molecules from pyscf to netket format. The support for PyScf is still experimental, and can be found in [Fermions and PyScf](experimental-fermions-api)

### Continuous space operators

This is a list of operators that you can define on {class}`~netket.experimental.hilbert.ContinuousHilbert` spaces.

```{eval-rst}
.. currentmodule:: netket.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   KineticEnergy
   PotentialEnergy
```

### Composing different operators together

Operators of different types, but acting on the same Hilbert space, can be combined by means of the operators described below. This is also useful to parametrize in a jax-friendly way time-dependent Hamiltonians.

```{eval-rst}
.. currentmodule:: netket.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   SumOperator
```


## Pre-defined operators

Those are easy-to-use constructors for a {class}`~netket.operator.LocalOperator`.

```{eval-rst}
.. currentmodule:: netket.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   boson.create
   boson.destroy
   boson.identity
   boson.number
   boson.proj
   spin.identity
   spin.sigmax
   spin.sigmay
   spin.sigmaz
   spin.sigmap
   spin.sigmam

```
