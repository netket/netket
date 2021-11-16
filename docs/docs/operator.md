(Hilbert)=
# The Operator module

```{eval-rst}
.. currentmodule:: netket.operator
```

The [Operator](`netket.operator`) module defines the common interfaces used to interact with quantum operators and super-operators, as well as several concrete implementations of different operators such as {ref}`netket.hilbert.LocalOperator`, {ref}`netket.hilbert.Ising` and others.

NetKet's operators are all sub-classes of the abstract class {ref}`netket.hilbert.AbstractOperator`, which defines a small set of API respected by all implementations. 
The inheritance diagram for the class hierarchy of the Operators included with NetKet is shown below (you can click on the nodes in the graph to go to their API documentation page). 
Dashed nodes represent abstract classes that cannot be instantiated, while the others are concrete and they can be instantiated.

```{eval-rst}
.. inheritance-diagram:: netket.operator
	:top-classes: netket.operator.AbstractOperator
	:parts: 1

```

Similarly to Hilbert spaces, there are two large classes of operators: {ref}`DiscreteOperator` and {ref}`ContinuousOperator`. 
Evidently the formers will only work with Discrete Hilbert spaces, while the latters will only work with continuous Hilbert spaces.

All {ref}`AbstractOperator`s act on a well defined hilbert space that can be accessed through the {attr}`~netket.hilbert.AbstractOperator.hilbert` attribute.
Compared to {ref}`netket.hilbert.AbstractHilbert` this interface is much simpler, exposing 

Hilbert space objects are all sub-classes of the abstract class {ref}`netket.hilbert.AbstractHilbert`, which defines the general API respected by all implementations. 
You can see a birds-eye of the inheritance diagram among the various kinds of Hilbert spaces included with NetKet below (you can click on the nodes in the graph to go to their API documentation page).


{ref}`netket.hilbert.AbstractHilbert` makes very few assumptions on the structure of the resulting space and you will generally very rarely interact with it directly.

There are then two more abstract Hilbert space types: {ref}`netket.hilbert.DiscreteHilbert`, representing Hilbert spaces where the local degrees of freedom are countable, and {ref}`netket.hilbert.ContinuousHilbert`, representing the Hilbert spaces with continuous bases, such as particles in a box. 

Those two abstract types are very different: `ContinuousHilbert` spaces are still experimental and we don't support yet many ways to manipulate them, while `DiscreteHilbert` spaces are much more developed and offer many utilities and handy functionalities.

The most important class of discrete Hilbert spaces are subclasses of {ref}`netket.hilbert.HomogeneousHilbert`, which is a space where the local degrees of freedom are identical among different sites. These subclasses are {ref}`netket.hilbert.Fock`, {ref}`netket.hilbert.Spin`, and {ref}`netket.hilbert.Qubit`.

{ref}`netket.hilbert.TensorHilbert` represents tensor products of different homogeneous hilbert spaces, therefore it is not homogeneous. You can use it to represent composite systems such as spin-boson setups.

{ref}`netket.hilbert.DoubledHilbert` represents a space doubled through [Choi's Isomorphism](https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism).
This is the space of density matrices and is used to work with dissipative/open systems.

## The `AbstractHilbert` interface
