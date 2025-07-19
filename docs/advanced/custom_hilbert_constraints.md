# Custom Hilbert Space Constraints

```{eval-rst}
.. currentmodule:: netket.hilbert
```

This guide explains how to implement custom constraints for discrete Hilbert spaces in NetKet, extending beyond the built-in constraints.


## Built-in Constraints

NetKet provides several built-in constraints for common use cases:

| Hilbert Space | Constraint Parameter | Implementation | Description | Example |
|---------------|---------------------|----------------|-------------|---------|
| {class}`Spin` | `total_sz` | {class}`~constraint.SumConstraint` | Fixed total magnetization $\sum_i \sigma_i$ | `Spin(0.5, N=4, total_sz=0)` |
| {class}`Fock` | `n_particles` | {class}`~constraint.SumConstraint` | Fixed total particle number $\sum_i n_i$ | `Fock(n_max=3, N=4, n_particles=2)` |
| {class}`SpinOrbitalFermions` | `n_fermions_per_spin` | {class}`~constraint.SumOnPartitionConstraint` | Fixed fermion number per spin subspace | `SpinOrbitalFermions(n_orbitals=4, s=0.5, n_fermions_per_spin=(2, 1))` |


We also ship the following extra classes:

- {class}`~constraint.DiscreteHilbertConstraint` - Base class for implementing custom constraints
- {class}`~constraint.ExtraConstraint` - Combine two constraints using logical AND

For more complex constraints or combinations of conditions, you'll need to implement custom constraints as described below.


```{admonition} Do you need custom constraints?

Constraints are used in 2 places throughout NetKet. 
1. To enumerate the full Hilbert Space, therefore when converting a NetKet operator to a numpy dense or scipy sparse matrix to perform exact diagonalisation.
2. To generate valid random configurations belonging to the subspace, used when initialising Markov Chain samplers.

In most serious VMC calculations, point 1 is not relevant because the system is too large to do ED anyway, and so you only really need 2.
However there are other, simpler methods of initialising MCMC samplers. For example, you could initialise them manually with a few valid configurations.

__So, while it's generally useful to implement your own constraints, you might not really need to do it.__
```

## Overview

Constraints in NetKet allow you to restrict the basis states of a Hilbert space to a subset that satisfies certain conditions. 
This is in general used to restrict the Hilbert spac to a projective subspace, which usually corresponds to a projective simmtry.

Note that constraints implicitly impose that your Hamiltonian is defined in the projected subspace of the constraint. If your Hamiltonian is not constrained to a subspace (e.g. an Hamiltonian which does not preserve the total number of excitations) then you cannot use a constraint.

```{admonition} Warning: Sampler Compatibility
:class: warning

When using constrained Hilbert spaces with Markov-Chain samplers, constraints guarantee that the initial state respects the constraint. However, **transition rules are not automatically constraint-aware.**

For example, {class}`~netket.sampler.MetropolisLocal` may violate particle number conservation. Use {class}`~netket.sampler.MetropolisExchange` for number-conserving dynamics, or implement custom transition rules for exotic constraints.
```

## Interface Requirements

To implement a custom constraint, you need:

To work with a custom constraint, you must do 2 things:
1. Define a **custom constraint class**, used to specify whether a configuration is valid or not. This must be a callable class inheriting from {class}`~constraint.DiscreteHilbertConstraint` that if passed a set of configurations will return an array of boolean flags telling netket whether those configurations are valid or not.
2. Optionally define an optimised custom {func}`random.random_state` dispatch rule specifying how to generate random configurations directly within the subspace. This is not needed, but the default fallback random state generation rule might be extremely slow for very constraining constraints. In principle this should return configurations distributed uniformly, but it is not terribly important (this is used to start the samplers, so even if it's a constant it might lead to worse warmup time but it might still work). 

### Defining a custom Constraint Class

Here we discuss how to Implement the **Custom Constraint Class**

Your constraint class must satisfy these requirements:

1. **Inherit from the base class**: Must inherit from {class}`~constraint.DiscreteHilbertConstraint`
2. **Be a valid NetKet PyTree dataclass**: Use {class}`~netket.utils.struct` fields with `pytree_node=False` for all attributes (constraints should have only static/non-trainable parameters)
3. **Implement required methods**:
   - `__call__(self, x)`: Takes a `(..., N)` shaped array of configurations and returns a `(...)` shaped boolean array indicating validity
   - `__hash__(self)`: For proper caching and JAX compilation
   - `__eq__(self, other)`: For constraint comparison
4. **Be JAX-compatible**: The `__call__` method must be {func}`jax.jit`-able

**Input/Output specification for `__call__`**:
- **Input**: `x` with shape `(..., N)` where `N` is the Hilbert space size and `...` represents arbitrary batch dimensions
- **Output**: Boolean array with shape `(...)` indicating whether each configuration satisfies the constraint

### Implementation Example

Here's a complete example implementing a sum constraint for Fock spaces:

```python
import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.utils import struct


class SumConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    """Constraint enforcing that the sum of all occupation numbers equals a target value."""

    total_sum: float = struct.field(pytree_node=False)

    def __init__(self, total_sum):
        self.total_sum = total_sum

    def __call__(self, x):
        """Check if configurations satisfy the constraint."""
        return jnp.sum(x, axis=-1) == self.total_sum

    def __hash__(self):
        return hash(("SumConstraint", self.total_sum))

    def __eq__(self, other):
        if isinstance(other, SumConstraint):
            return self.total_sum == other.total_sum
        return False


# Usage
hilbert = nk.hilbert.Fock(n_max=3, N=4, constraint=SumConstraint(5))
print(f"Constrained space has {hilbert.n_states} states")

# Verify constraint
states = hilbert.all_states()
print(f"All states sum to {hilbert.constraint.total_sum}: {jnp.all(jnp.sum(states, axis=1) == 5)}")
```

### Optimized Random State Generation

For better performance, especially during sampling initialization, implement a custom random state generator:

```python
@nk.hilbert.random.random_state.dispatch
def random_state_sumconstraint(
    hilb: nk.hilbert.Fock, 
    constraint: SumConstraint, 
    key, 
    batches: int, 
    *, 
    dtype=None
):
    """Generate random states satisfying the sum constraint."""
    dtype = jnp.result_type(hilbert._local_states.dtype, dtype)

    def random_constraints_py(key_data):
        # Use numpy for efficient constraint-satisfying generation
        rng = np.random.default_rng(np.array(key_data))
        
        states = np.zeros((batches, hilb.size), dtype=dtype)
        
        for i in range(batches):
            # Simple algorithm: distribute particles randomly
            remaining = constraint.total_sum
            for j in range(hilb.size - 1):
                # Random number of particles at this site
                max_here = min(remaining, hilb.shape[j] - 1)
                n_here = rng.integers(0, max_here + 1)
                states[i, j] = n_here
                remaining -= n_here
            states[i, -1] = remaining  # Put rest at last site
            
            # Randomly shuffle to avoid bias toward last sites
            rng.shuffle(states[i])
            
        return states

    return jax.pure_callback(
        random_constraints_py,
        jax.ShapeDtypeStruct((batches, hilb.size), dtype=dtype),
        jax.random.key_data(key),
    )


# Test the optimized generator
key = jax.random.PRNGKey(42)
states = hilbert.random_state(key, 1000)
print(f"Generated states all satisfy constraint: {jnp.all(jnp.sum(states, axis=1) == 5)}")
```

## Advanced Patterns

### Complex Constraints

For more sophisticated constraints involving multiple conditions:

```python
class ParityAndSumConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    """Enforce both sum and parity constraints."""
    
    total_sum: int = struct.field(pytree_node=False)
    even_parity: bool = struct.field(pytree_node=False)

    def __init__(self, total_sum, even_parity=True):
        self.total_sum = total_sum
        self.even_parity = even_parity

    def __call__(self, x):
        sum_condition = jnp.sum(x, axis=-1) == self.total_sum
        
        # Count occupied sites
            
        if self.even_parity:
            parity_condition = occupied % 2 == 0
        else:
            parity_condition = occupied % 2 == 1
            
        return sum_condition & parity_condition

    def __hash__(self):
        return hash(("ParityAndSumConstraint", self.total_sum, self.even_parity))

    def __eq__(self, other):
        if isinstance(other, ParityAndSumConstraint):
            return (self.total_sum == other.total_sum and 
                   self.even_parity == other.even_parity)
        return False

# Usage
hilbert = nk.hilbert.Fock(n_max=3, N=4, 
    constraint=ParityAndSumConstraint(4, True))
print(f"Constrained space has {hilbert.n_states} states")

# Verify constraint
states = hilbert.all_states()

```

### Using Pure Callbacks for Complex Logic

When JAX-native implementation is difficult, use {func}`jax.pure_callback`:

```python
def complex_constraint_check(configurations):
    """Complex constraint logic in pure Python/NumPy."""
    # Your complex logic here
    return np.array([check_single_config(config) for config in configurations])

class ComplexConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    def __call__(self, x):
        return jax.pure_callback(
            complex_constraint_check,
            jax.ShapeDtypeStruct(x.shape[:-1], bool),
            x,
            vmap_method="expand_dims"
        )
```

## Integration with Variational States

Constrained Hilbert spaces work seamlessly with all NetKet variational states:

```python
# Create constrained system
hilbert = nk.hilbert.Spin(0.5, N=10, constraint=SumConstraint(0))  # Zero magnetization
graph = nk.graph.Chain(10)
hamiltonian = nk.operator.Ising(hilbert, graph, h=1.0)

# Standard NetKet workflow
model = nk.models.RBM(alpha=1)
sampler = nk.sampler.MetropolisExchange(hilbert, graph)  # Respects magnetization
vstate = nk.vqs.MCState(sampler, model)

# Optimization works as usual
optimizer = nk.optimizer.Sgd(0.01)
driver = nk.driver.VMC(hamiltonian, optimizer)
driver.run(vstate, n_iter=100)
```

## See Also

- {doc}`../user-guides/hilbert` - Basic Hilbert space usage
- {class}`~netket.hilbert.constraint.DiscreteHilbertConstraint` - Base constraint class API
- {func}`~netket.hilbert.random.random_state` - Random state generation interface
- {doc}`../user-guides/sampler` - Sampler compatibility considerations