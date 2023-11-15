# Defining Custom Preconditioners

NetKet calls _gradient preconditioner_ that class of techniques that transform the
gradient of the cost function in order to improve convergence properties before
passing it to an optimiser like {func}`netket.optimizer.Sgd` or {func}`netket.optimizer.Adam`.

Examples of _gradient preconditioners_ are the [Stochastic Reconfiguration (SR)](https://www.attaccalite.com/PhDThesis/html/node15.html) method  (also known as [Natural Gradient Descent](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) in ML literature), the Linear Method or second order Hessian-based optimisation.

We call those methods _gradient preconditioners_ because they take as input the gradient of the cost function (e.g., the energy gradient in VMC ground state optimisation) and output a transformed gradient.

In the current version, NetKet provides Stochastic Reconfiguration ({func}`netket.optimizer.SR`) as a built-in method.
It is also possible to define your own method. If you implement the API as outlined in this document, you will be able to use your own preconditioner for use in NetKet optimisation driver without issues.

Keep in mind that writing your own optimisation loop only requires writing about 10  lines of code and you are not forced to use NetKet's drivers!
We believe our design to be fairly modular and flexible and thus should be able to accommodate a wide variety of use-cases, but there will be algorithms that are hard to express within the boundaries of our API.
Do not turn away from NetKet in those cases, but take all the pieces that you need and write your own optimisation loop!

## The preconditioner interface

Preconditioners must be implemented as a Callable object or function with the following signature:

```python
def preconditioner(vstate: VariationalState, gradient: PyTree) -> PyTree:
```

The Callable must accept two (positional) inputs, where the first is the variational state itself and the latter is the current gradient, stored as a PyTree.
The output of the preconditioner must be the transformed gradient stored as a PyTree.

This general API will allow you to implement any preconditioner and use it together with NetKet's variational optimization drivers.
However, note that any performance optimisation (such as calling {func}`jax.jit` on the code) will be your responsibility.

### The LinearPreconditioner interface

Several preconditioners, including the Stochastic Reconfiguration, transform the gradient by solving a linear system of equation.

$$
S \bf{x} + \bf{F}
$$

where $ S $ is a linear operator, $ F $ is the gradient and the solution $\bf{x}$ is the preconditioned gradient.

NetKet implements a basic interface called {class}`netket.optimizer.LinearPreconditioner` to make it easier to implement this kind of
solvers. It is especially tuned for the cases where $ S $ is a linear operator.

To construct {class}`netket.optimizer.LinearPreconditioner` you must supply two objects: the `lhs_constructor`, which is a function or
closure with signature `(VariationalState)->LinearOperator` that accepts one argument, the variational state, and constructs the linear
operator associated with it.
The other object is a linear solver method, that must accept the linear operator and the gradient and compute the solution.
The gradient is always provided as a PyTree.

To give a clear example: in the case of the Stochastic Reconfiguration (SR) method, if we call $ \bf{F} $ the gradient of the energy and $ S $ the Quantum Geometric Tensor (also known as SR matrix), we need to solve the system of equation $ S d\bf{w} = F $ to compute the resulting gradient.
The $ S $ matrix in this case is the `lhs` or LinearOperator of the preconditioner, while the function is any linear solver such as `cholesky`, {func}`jax.numpy.linalg.solve` or iterative solvers such as {func}`jax.scipy.sparse.linalg.cg`.

As there are different ways to compute the $ S $ matrix, all with their different computational performance characteristics, and there are different solvers, we believe that this design makes the code more modular and easier to reason about.

When defining a preconditioner object you have two options: you can implement the bare API, which gives you maximum freedom but makes you responsible for all optimisations, or you can implement the `~netket.optimizer.LinearOperator` interface, which constraints you a bit but will take care of a few performance optimisations.

#### Bare interface

The bare-minimum API a preconditioner `lhs` must implement:

    - It must be a class

    - There must be a function to build it from a variational state. This function will be called with the variational state as the first positional argument.  This function must not necessarily be a method of the class.

    - This class must have a `solve(self, function, gradient, *, x0=None)` method taking as argument the gradient to be preconditioned and must not error if a keyword argument `x0` is passed to it. `x0` is the output of `solve` the last time it has been called, and might be ignored if not needed. `function` is the function computing the preconditioner.

You can subclass the abstract base class {class}`netket.optimizer.PreconditionerObject` to be sure that you are
implementing the correct interface, but you are not obliged to subclass it.

When you implement such an interface you are left with maximum flexibility, however you will be responsible for `jax.jit`ing all computational intensive methods (most likely `solve`).

```python
def MyObject(variational_state):
    stuff_a, stuff_b = compute_stuff(variational_state)
    return MyObjectT(stuff_a, stuff_b)


class MyObjectT:
    def __init__(self, a,b):
        # setup this object

    def solve(self, preconditioner_function, y, *, x0=None):
        # prepare
        ...
        # compute
        return solve_fun(self, y, x0=x0)
```

Be warned that if you want to {func}`jax.jit` compile the solve method, as it is usually computationally intensive, you must either specify how to flatten and unflatten to a PyTree your `MyObjectT`, or you should mark it as a `flax.struct.dataclass`, which is a frozen dataclass which does that automatically.
Since you cannot write the `__init__` method for a frozen dataclass, we usually define a constructor function as shown above.

You might be asking why each object needs a `solve` method and is passed to the preconditioner function instead of the over way around. The reason for this is to invert the control: `preconditioner_function`s must obey a certain API, but even if they do, different objects might need to perform some different initialization to compute the precondition in a more efficient way.
This architecture allows every object to run arbitrary logic before executing the preconditioner of choice.
Particular examples of this approach can be seen by looking at the implementation of {func}`netket.optimizer.qgt.QGTJacobianDense` and {func}`netket.optimizer.qgt.QGTJacobianPyTree`.


#### LinearOperator interface

You can also subclass {class}`~netket.optimizer.LinearOperator`.
A LinearOperator must be a [`flax` dataclass](https://flax.readthedocs.io/en/latest/flax.struct.html), which is an immutable
object (therefore after construction you cannot modify its attributes).

LinearOperators have several convenience methods, and they will act as matrices: you can right-multiply them by a PyTree
vector or a dense vector. You can obtain their dense representation, and it will automatically jit-compile all computationally
intensive operations.

To implement the LinearOperator interface you should implement the following methods:

```python

def MyLinearOperator(variational_state):
    stuff_a, stuff_b = compute_stuff(variational_state)
    return MyLinearOperatorT(stuff_a, stuff_b)

@flax.struct.dataclass
class MyLinearOperatorT(nk.optimizer.LinearOperator):

    @jax.jit
    def __matmul__(self, y):
        # prepare
        ...
        # compute
        return result

    @jax.jit
    def to_dense(self):
        ...
        return dense_matrix

    #optional
    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
        #...
        return solution, solution_info
```

The bare minimum thing to implement is `__matmul__`, specifying how to multiply the linear operator by a pytree.
You can also define a custom `_solve` method if you have some computationally intensive setup code you wish to
run before executing a solve function (that will call matmul repeatedly).
The `_solve` takes as first input the solve function, which is passed as a closure so it does not need to be marked
as static (even though it is).  The x0 is an optional argument which must be accepted but can be ignored, and it is the last previous solution to the linear system.
Optionally, one can also define the `to_dense` method.


### The preconditioner function API

The preconditioner function must have the following signature:

```python
def preconditioner_function(object, gradient, *, x0=None):
    #...
    return preconditioned_gradient, x0
```

The object that will be passed is the selected preconditioner object, previously constructed.
The gradient is the gradient of the loss function to precondition.
x0 is an optional initial condition that might be ignored.

The gradient might be a PyTree version or a dense ravelling of the PyTree. The result of the function should be a preconditioned gradient with the same format.
Additional keyword argument can be present, and will in general be set through a closure or `functools.partial`, because this function will be called with the signature above.
If you have a peculiar preconditioner, you can assume that `preconditioner_function` will be called only from your `preconditioner object`, but in general it is good practice respecting the interface above so that different functions can work with different objects.


