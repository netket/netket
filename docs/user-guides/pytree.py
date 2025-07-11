# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %% tags=["hide-input", "hide-output"]
# Try to load netket, and install it if the import fails
try:
    import netket as nk
except ImportError:
    # !pip install --quiet --upgrade netket
    
import netket as nk
import jax
import jax.numpy as jnp

from netket.utils.struct import Pytree, field, static_field
from flax import serialization
import dataclasses

# %%

# %% [markdown]
# (Pytree)=
# # NetKet Pytrees
#
# ```{eval-rst}
# .. currentmodule:: netket.utils.struct
# ```
#
# The `Pytree` class provides the foundation for creating JAX-compatible 
# data structures that can be seamlessly used with JAX transformations 
# like `jax.jit`, `jax.vmap`, and `jax.grad`. 
#
# A PyTree in JAX is a tree-like data structure composed of containers (like tuples, lists, or dictionaries) with leaves that are arrays or scalars. The `Pytree` class in NetKet extends this concept to custom classes, allowing them to be transparently used with JAX transformations while maintaining object-oriented design principles.
#
# The `Pytree` class distinguishes between two types of fields:
#
# - **Dynamic fields (PyTree nodes)**: These are part of the computational graph and can be transformed by JAX. They typically contain arrays, parameters, or other data that changes during computation.
# - **Static fields**: These are metadata or configuration parameters that remain constant during JAX transformations. They must be hashable and are excluded from differentiation.
#
# ## Basic Usage
#

# %%
from netket.utils import struct

class SimpleData(struct.Pytree):
    """A simple data container demonstrating basic Pytree usage."""
    
    # Dynamic field - will be part of JAX transformations
    values: jax.Array
    
    # Static field - configuration that doesn't change during computation
    size: int = struct.static_field()
    
    def __init__(self, values, size):
        self.values = values
        self.size = size
    
    def sum(self):
        return jnp.sum(self.values)

# Create an instance
data = SimpleData(
    values=jnp.array([1.0, 2.0, 3.0]),
    size=3
)

print(f"Data values: {data.values}")
print(f"Data size: {data.size}")
print(f"Sum: {data.sum()}")

# %%
# The data can be used directly with JAX transformations
@jax.jit
def compute_mean(data):
    return jnp.mean(data.values)

result = compute_mean(data)
print(f"Mean: {result}")

# %%
# When we inspect the PyTree structure, we see only dynamic fields
leaves = jax.tree.leaves(data)
print(f"PyTree leaves: {leaves}")

# Static fields are preserved during transformations
transformed_data = jax.tree.map(lambda x: x * 2, data)
print(f"Transformed values: {transformed_data.values}")
print(f"Preserved size: {transformed_data.size}")

# %% [markdown]
# ## Features
#
# ### Immutability and the `replace` Method
#
# By default, `Pytree` objects are immutable, similar to frozen dataclasses. This immutability is crucial for JAX's functional programming paradigm and ensures that transformations don't have unexpected side effects.

# %%
# Trying to modify a field directly will raise an error
try:
    data.values = jnp.array([4.0, 5.0, 6.0])
except AttributeError as e:
    print(f"Error: {e}")

# %%
# Instead, use the replace method to create a new instance with modified values
new_data = data.replace(values=jnp.array([4.0, 5.0, 6.0]))
print(f"Original values: {data.values}")
print(f"New values: {new_data.values}")
print(f"Size unchanged: {new_data.size}")

# %% [markdown]
# ### Mutable PyTrees
#
# While immutability is the default and recommended approach, you can create mutable PyTrees when needed. This is particularly useful during development or when working with algorithms that require in-place modifications.

# %%
class Counter(Pytree, mutable=True):
    """A simple mutable counter."""
    
    count: jax.Array
    step_size: int = static_field(default=1)
    
    def __init__(self, count, step_size=1):
        self.count = count
        self.step_size = step_size
    
    def increment(self):
        """Increment counter in-place."""
        self.count = self.count + self.step_size

# Create a mutable counter
counter = Counter(
    count=jnp.array(0),
    step_size=2
)

print(f"Initial count: {counter.count}")

# Update counter in-place
counter.increment()
print(f"After increment: {counter.count}")

# %% [markdown]
# ### Field Types and Metadata
#
# The `field` function provides fine-grained control over how fields are handled in PyTrees. It supports various options for serialization, caching, and distributed computing.

# %%
class Record(Pytree):
    """Demonstrates advanced field configurations."""
    
    # Standard dynamic field
    data: jax.Array
    
    # Static field with default value
    name: str = static_field(default="default")
    
    # Field with custom serialization name
    info: jax.Array = field(serialize_name="information")
    
    # Field that won't be serialized
    temp: jax.Array = field(serialize=False)
    
    # Field with default factory
    metadata: dict = field(default_factory=dict, pytree_node=False)
    
    def __init__(self, data, info, temp=None, name="default"):
        self.data = data
        self.info = info
        self.temp = temp if temp is not None else jnp.zeros(2)
        self.name = name
        self.metadata = {"created": True}

record = Record(
    data=jnp.array([1.0, 2.0, 3.0]),
    info=jnp.array([0.1, 0.2]),
    name="example"
)

print(f"Name: {record.name}")
print(f"Metadata: {record.metadata}")

# %% [markdown]
# ### Serialization with Flax
#
# `Pytree` objects integrate seamlessly with Flax's serialization system, allowing you to save and load object states efficiently. This is particularly important for checkpointing and data persistence.

# %%
# Serialize the record to a state dictionary
state_dict = serialization.to_state_dict(record)
print("Serialized state dictionary:")
for key, value in state_dict.items():
    print(f"  {key}: {value}")

# %%
# Notice that temp is not serialized (serialize=False)
# and info is stored under "information" (serialize_name="information")

# Create a new record instance with different values
new_record = Record(
    data=jnp.zeros(3),
    info=jnp.zeros(2),
    temp=jnp.ones(2)
)

# Restore from the state dictionary
restored_record = serialization.from_state_dict(new_record, state_dict)

print(f"Restored data: {restored_record.data}")
print(f"Restored info: {restored_record.info}")
print(f"Temp (not restored): {restored_record.temp}")

# %% [markdown]
# ### Dynamic Node Discovery
#
# For maximum flexibility, you can enable dynamic node discovery, which allows fields to be added at runtime. This is useful when the structure of your PyTree depends on runtime conditions.

# %%
class FlexibleData(Pytree, dynamic_nodes=True):
    """A data structure that can have fields added dynamically."""
    
    base: jax.Array
    config: str = static_field(default="default")
    
    def __init__(self, base, config="default", **kwargs):
        self.base = base
        self.config = config
        
        # Add additional fields dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

# Create data with dynamic fields
flex_data = FlexibleData(
    base=jnp.array([1.0, 2.0]),
    config="custom",
    extra1=jnp.array([3.0, 4.0, 5.0]),
    extra2=jnp.array([6.0, 7.0])
)

print(f"Base: {flex_data.base}")
print(f"Extra1: {flex_data.extra1}")
print(f"Extra2: {flex_data.extra2}")

# All dynamic fields become part of the PyTree
leaves = jax.tree_util.tree_leaves(flex_data)
print(f"Number of PyTree leaves: {len(leaves)}")

# %% [markdown]
# ### Inheritance and Composition
#
# `Pytree` classes can be inherited and composed to build complex hierarchical structures. This is useful for creating modular, reusable components.

# %%
class BaseContainer(Pytree):
    """Base class for data containers."""
    
    data: jax.Array
    label: str = static_field(default="base")
    
    def __init__(self, data, label="base"):
        self.data = data
        self.label = label

class NumberContainer(BaseContainer):
    """Container for numbers with additional operations."""
    
    scale: float = static_field(default=1.0)
    
    def __init__(self, data, label="numbers", scale=1.0):
        super().__init__(data, label)
        self.scale = scale
    
    def scaled_sum(self):
        return jnp.sum(self.data) * self.scale

# Create a number container
numbers = NumberContainer(
    data=jnp.array([1.0, 2.0, 3.0]),
    label="test",
    scale=2.0
)

print(f"Data: {numbers.data}")
print(f"Label: {numbers.label}")
print(f"Scale: {numbers.scale}")
print(f"Scaled sum: {numbers.scaled_sum()}")

# %% [markdown]
# ## Working with JAX Transformations
#
# The real power of `Pytree` becomes apparent when working with JAX transformations. Let's explore how PyTrees behave under various JAX operations.

# %%
class OptimizableData(Pytree):
    """A data structure suitable for optimization."""
    
    values: jax.Array
    target: jax.Array
    learning_rate: float = static_field(default=0.01)
    
    def __init__(self, values, target, learning_rate=0.01):
        self.values = values
        self.target = target
        self.learning_rate = learning_rate
    
    def loss(self):
        return jnp.mean((self.values - self.target) ** 2)

# Create optimizable data
opt_data = OptimizableData(
    values=jnp.array([1.0, 2.0, 3.0]),
    target=jnp.array([1.5, 2.5, 3.5])
)

print(f"Initial loss: {opt_data.loss()}")

# %%
# Compute gradients with respect to the data
@jax.jit
def compute_gradients(data):
    return jax.grad(lambda d: d.loss())(data)

gradients = compute_gradients(opt_data)
print(f"Gradients: {gradients.values}")

# Note: static fields like learning_rate are not differentiated
print(f"Learning rate unchanged: {gradients.learning_rate}")

# %%
# Update the data using gradients
@jax.jit
def update_data(data, gradients):
    return data.replace(
        values=data.values - data.learning_rate * gradients.values
    )

updated_data = update_data(opt_data, gradients)
print(f"Updated values: {updated_data.values}")
print(f"Updated loss: {updated_data.loss()}")

# %% [markdown]
# ## Real-World Example: Sampler State
#
# Let's look at how `Pytree` is used in NetKet's sampler module to manage the state of Monte Carlo samplers. This example shows the practical application of PyTrees in a complex quantum simulation context.

# %%
class SimpleSamplerState(Pytree):
    """Simplified version of NetKet's sampler state."""
    
    # Current configurations (dynamic - part of computation)
    configurations: jax.Array
    
    # Log probabilities (dynamic - computed values)
    log_probs: jax.Array
    
    # Random number generator state (dynamic - changes during sampling)
    rng_state: jax.Array
    
    # Number of accepted moves (dynamic - statistics)
    n_accepted: jax.Array
    
    # Number of steps taken (dynamic - statistics)
    n_steps: jax.Array
    
    # Sampler configuration (static - doesn't change during sampling)
    n_chains: int = static_field()
    sweep_size: int = static_field()
    
    def __init__(self, configurations, log_probs, rng_state, n_chains, sweep_size):
        self.configurations = configurations
        self.log_probs = log_probs
        self.rng_state = rng_state
        self.n_accepted = jnp.zeros(n_chains, dtype=int)
        self.n_steps = jnp.zeros((), dtype=int)
        self.n_chains = n_chains
        self.sweep_size = sweep_size
    
    @property
    def acceptance_rate(self):
        """Compute the acceptance rate."""
        return jnp.mean(self.n_accepted) / (self.n_steps + 1e-10)

# Create a sampler state
key = jax.random.PRNGKey(42)
sampler_state = SimpleSamplerState(
    configurations=jax.random.normal(key, (4, 10)),  # 4 chains, 10 sites
    log_probs=jnp.array([-1.0, -2.0, -1.5, -1.8]),
    rng_state=jax.random.PRNGKey(123),
    n_chains=4,
    sweep_size=10
)

print(f"Configurations shape: {sampler_state.configurations.shape}")
print(f"Initial acceptance rate: {sampler_state.acceptance_rate}")

# %%
# Simulate a sampling step
@jax.jit
def sampling_step(state):
    """Simulate a single sampling step."""
    # Split the RNG key
    new_key, subkey = jax.random.split(state.rng_state)
    
    # Simulate some accepted moves
    new_accepted = state.n_accepted + jax.random.bernoulli(subkey, 0.3, shape=(state.n_chains,)).astype(int)
    new_steps = state.n_steps + 1
    
    # Update some configurations
    new_configs = state.configurations + jax.random.normal(subkey, state.configurations.shape) * 0.1
    
    return state.replace(
        configurations=new_configs,
        n_accepted=new_accepted,
        n_steps=new_steps,
        rng_state=new_key
    )

# Run several sampling steps
current_state = sampler_state
for i in range(3):
    current_state = sampling_step(current_state)
    print(f"Step {i+1}: Acceptance rate = {current_state.acceptance_rate:.3f}")
