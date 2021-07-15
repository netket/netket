# Overriding behaviour and defining custom operators

```{currentmodule} netket
```

In this section we will discuss how it is possible to extend NetKet or override
default behaviour.
NetKet is architectured in such a way that an user can override several parts of
its default behaviour without having to edit its source code.
You can simply fire up a jupyter instance, define a few methods overloading 
the default, and you're all set. 
This is also valid if you want to define a custom object, such as a custom 
hilbert space or operator. 


## NetKet Architecture: Multiple Dispatch

Some parts of NetKet rely on multiple-dispatch in order to select the right implementation
of a function, instead of the more limited single dispatch used in 
traditional Object-Oriented programming.
Significant inspiration has been taken from the [Julia programming language](https://julialang.org/), 
and NetKet relies on an somehwat-faithfull implementation of Julia's dispatch into python, 
provided by the [plum-dispatch](https://github.com/wesselb/plum) package.

If you are already familiar with multiple-dispatch, you can skip this section. 
If you want to see how to use multiple-dispatch in python, we refer you to the plum 
documentation. Otherwise, read on!

The rationale behind multiple dispatch is the following: If you have a function acting on 
two objects, such as the `expect` function computing the expectation value of an operator
over a variational state, how do you make sure that `expect` works for any combination of
different variational states and operators?

The standard object-oriented approach is to use single-dispatch to dispatch on the first 
argument (or owner) of the method. Therefore if `expect` is a method of `VariationalState`,
you would be writing an `expect` method for every different `VariationalState`. This method
should work with any operator you throw at it.

Maybe you are capable of defining a very broad interface for Operators so that your implementation
works for all the operators already defined, but how can an user define a new `CrazyOperator` that
does not follow this convention?

Multiple Dispatch makes it possible to define a method of a function that is only used when _all_
the types match a given signature, and as long as the return type of the function makes sense, 
it will work with the rest of NetKet.  If my expect function is defined as:

```python
@netket.utils.dispatch.dispatch
def expect(vstate : VariationalState, operator: AbstractOperator):
    # a generic algorithm that works for any operator in NetKet
    return netket.stats.statistics(result)
```

And this function works for anything already implemented in Netket and always returns a 
statistics object (`netket.stats.Stat`). 
However, imagine that my `CrazyOperator` is defined on a Crazy Hilbert space that always 
gives random results. This is clearly incompatible with the standard implementation. 
However, instead of figuring a way to make such `CrazyOperator` fit in the what already exists,
we can just define a custom dispatch rule as follows:

```python
class CrazyOperator:
    """ A really crazy implementation. """
    sigma = 0.1

@expect.dispatch
def expect(vstate : MCState, operator: CrazyOperator):
    # A crazy implementation that only works for CrazyOperator
    return netket.stats.statistics(np.random.rand(100)*operator.sigma)

```

And everything will work as expected.
Of course, to make everything _really_ work, it would be best to have `CrazyOperator` 
subclass `AbstractOperator`, but that is not needed if you are carefull enough.

The types that determine the dispatch are picked up by the type hints.


## Defining custom implementations for expectation values

```{eval-rst}
.. autofunction:: netket.vqs.expect
```

If you want to overload the default behaviour, or define those operations for 
your custom types, you must define dispatch rules for the two methods listed above.

The first, `expect`, is used when you call `VariationalState.expect`, while the latter
is used when computing the expectation value and the gradient.

You should be carefull in returning the right signature: `expect` takes as input the
variational state and operator and must return a number or a `nk.stats.Stats` object
with statistical information about it.

```{eval-rst}
.. autofunction:: netket.vqs.expect_and_grad
```

`expect_and_grad` has a more complex signature, and must return a tuple where the first
object is the same as `expect` and the latter is the gradient stored into a PyTree.

The `use_covariance` argument signals whever you should be using the covariance formula or
not. This argument is a `True/False` bool-like object that can be dispatched upon using
the types `netket.utils.dispatch.TrueT` and `netket.utils.dispatch.FalseT`.
By default, all hermitian operators will use the covariance formula.
You can chose to ignore this argument.