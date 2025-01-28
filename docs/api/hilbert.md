(netket_hilbert_api)=
# netket.hilbert

```{eval-rst}
.. currentmodule:: netket.hilbert

```

Hilbert-space objects determine the state space of a quantum system and a specific choice of basis. 
All implementations of Hilbert spaces derive from the class `AbstractHilbert` and fall into two classes:
 - discrete Hilbert spaces, which inherit from the abstract class `DiscreteHilbert` and include spin ({class}`Spin`), Fock ({class}`Fock`), and qubit ({class}`Qubit`) spaces. Discrete spaces are typically used to describe lattice systems. 
 The lattice structure itself is, however, not part of the Hilbert space class and can be defined separately.
 - continuous Hilbert spaces, which inherit from the abstract class {class}`ContinuousHilbert`. 
 Currently, the only concrete continuous space provided by NetKet is {class}`Particle`.


```{eval-rst}
.. inheritance-diagram:: netket.hilbert netket.experimental.hilbert
   :top-classes: netket.hilbert.AbstractHilbert
   :parts: 1

```

## Concrete Classes

Below you find a list of all concrete Hilbert spaces that you can use.

```{eval-rst}
.. currentmodule:: netket.hilbert

.. autosummary::
   :toctree: _generated/hilbert
   :template: class
   :nosignatures:

   TensorHilbert
   DoubledHilbert
   Spin
   Qubit
   Fock
   Particle
   SpinOrbitalFermions
```


## Abstract Classes

Below you find a list of all abstract classes defined in this module. 
Those classes cannot be directly instantiated, but you can inherit from one of them if you want to define new hilbert spaces.

```{eval-rst}
.. currentmodule:: netket.hilbert

.. autosummary::
   :toctree: _generated/hilbert
   :template: class
   :nosignatures:

   AbstractHilbert
   ContinuousHilbert
   DiscreteHilbert
   HomogeneousHilbert

```

### Constraints

```{eval-rst}
.. currentmodule:: netket.hilbert

.. autosummary::
   :toctree: _generated/hilbert
   :template: class
   :nosignatures:

   constraint.DiscreteHilbertConstraint
```

### Random submodule

When defining a new Hilbert space, you must define how to uniformly sample the basis elements of that Hilbert space by defining some dispatch rules for those functions.

```{eval-rst}
.. currentmodule:: netket.hilbert

.. autosummary::
   :toctree: _generated/hilbert
   :nosignatures:

   random.flip_state
   random.random_state
   index.optimalConstrainedHilbertindex
```

