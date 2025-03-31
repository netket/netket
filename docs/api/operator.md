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
   BoseHubbardJax
   GraphOperator
   LocalOperator
   LocalOperatorJax
   Ising
   IsingJax
   Heisenberg
   PauliStrings
   PauliStringsJax
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
   FermionOperator2ndJax
   fermion.create
   fermion.destroy
   fermion.number
```

Note in particular the pyscf module that can be used to convert molecules from pyscf to netket format. The support for PyScf is still experimental, and can be found in [Fermions and PyScf](experimental-fermions-api)

### Continuous space operators

This is a list of operators that you can define on {class}`~netket.hilbert.ContinuousHilbert` spaces.

```{eval-rst}
.. currentmodule:: netket.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   KineticEnergy
   PotentialEnergy
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

In the experimental submodule there are also easy-to-use constructors for common {class}`~netket.experimental.operator.FermionOperator2nd`.

```{eval-rst}
.. currentmodule:: netket

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   experimental.operator.fermion.create
   experimental.operator.fermion.destroy
   experimental.operator.fermion.identity
   experimental.operator.fermion.number
```

## QuSpin Integration

NetKet now includes methods to convert operators to the QuSpin format and target specific symmetry subsectors.

### Converting to QuSpin Format

You can convert a NetKet operator to the QuSpin format using the `to_quspin_format` function. Here's an example:

```python
from netket.operator import to_quspin_format
from netket.operator import Ising
from netket.hilbert import Spin

# Define a NetKet operator
hilbert = Spin(s=0.5, N=10)
operator = Ising(hilbert, h=1.0)

# Convert to QuSpin format
quspin_operator = to_quspin_format(operator)
```

### Targeting Symmetry Subsectors

You can target a specific symmetry subsector for a NetKet operator using the `target_symmetry_subsector` function. Here's an example:

```python
from netket.operator import target_symmetry_subsector
from netket.operator import Ising
from netket.hilbert import Spin

# Define a NetKet operator
hilbert = Spin(s=0.5, N=10)
operator = Ising(hilbert, h=1.0)

# Target a specific symmetry subsector
subsector_operator = target_symmetry_subsector(operator, subsector="translation")
```
