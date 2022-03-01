(Operator)=
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

The main function of operators in NetKet is to define the logic and some values used to compute expectation values over a variational state.
Functions implemented by operators are either needed to compute expectation values, or are nice utilities useful to manipulate or inspect the operators but are not needed by the Monte-Carlo logic interacting with the variational states.

All {ref}`AbstractOperator`s act on a well defined hilbert space that can be accessed through the {attr}`~netket.hilbert.AbstractOperator.hilbert` attribute.
It is also possible to check if the operator is hermitian through the boolean property {attr}`~netket.hilbert.AbstractOperator.is_hermitian`.
There are only two other operations defined on all operator types: it is possible to take the conjugate or conjugate-transpose of an operator by accessing the methods
{meth}`~netket.hilbert.AbstractOperator.conj` and {attr}`~netket.hilbert.AbstractOperator.transpose`. 
Those will usually return lazy wrappers.
Finally, it is also possible to call {meth}`~netket.hilbert.AbstractOperator.collect` to get rid of any possible lazy wrapper.

The bare-minimum requirement when defining a custom operator is to define it's hilbert space. 
Most likely you will also want to define the `expect` and/or the `expect_and_grad` method to compute the expectation value of such operator over a certain Variational State. 
Contrary to more standard Pythonic code, those methods are not defined as class-functions in your custom operator class, but you have to use multiple dispatch ({ref}`netket.utils.dispatch.dispatch`) to define those methods on a specific signature such as `expect(vstate: MCState, O: MyCustomOperator)`. 
This is needed because the way you compute expectation values and/or gradients might not only change depending on the exact operator, but depends also on the type of variational state that you are working with.
To learn more about multiple dispatch, check {ref}`this section <multiple-dispatch>`

An explanation of how to define the `expect` method for custom operators is given in the [Custom Operator documentation](../advanced/custom_operators).
