# Architecture & Philosophy

NetKet is designed to empower researchers to both explore new ideas in quantum many-body physics and run established Neural Quantum State (NQS) algorithms with ease. 
Understanding NetKet's modular architecture is key to effectively using and extending the library.


## Tiered Modular Architecture

NetKet is organized into three distinct tiers, each serving different needs and providing different levels of abstraction:

```{list-table} NetKet Architecture Overview
:header-rows: 1
:widths: 15 25 30 30

* - **Tier**
  - **Purpose**
  - **Usage**
  - **Characteristics**
* - **Tier 0**
  - Core JAX utilities
  - Algorithm implementation
  - JAX-native, reusable tools. Few changes over the years.
* - **Tier 1** 
  - VMC building blocks
  - Everyone, including those who want to write their own algorithms.
  - Extensible, composable and well-defined interfaces.
* - **Tier 2**
  - MCState interface & optimisation drivers
  - General users, beginners.
  - Opinionated, easy-to-use abstractions.
```

---

## Tier 0: Core JAX Tools

**Purpose**: Low-level utilities that make working with JAX more efficient, particularly for distributed computing and memory management.

These tools are JAX-native and could be used outside of NetKet. 
They are quite intricate and have seen few changes over the years, but are essential if you want to write code that efficiently runs across many GPUs while automatically maintaining a low memory consumption.

In general, if you want to write algorithms using NetKet (or not even) operators, you will have to use those tools.

```{list-table} Tier 0 Modules
:header-rows: 1
:widths: 30 70

* - **Module**
  - **Description**
* - {mod}`netket.jax`
  - JAX utilities: chunking, sharding, distributed operations. {func}`~netket.jax.vmap_chunked`, {func}`~netket.jax.vjp_chunked` and {func}`~netket.jax.jacobian`.
* - {mod}`netket.utils.struct`
  - Custom PyTree implementations and data structures. Virtually all NetKet classes are {class}`~netket.utils.struct.Pytree`s.
* - {mod}`netket.utils.timing`
  - Performance profiling and timing utilities
* - {mod}`netket.utils.numbers`
  - Numerical utilities and special number types
```

---

## Tier 1: VMC Building Blocks

Fundamental components needed to build Variational Monte Carlo algorithms with well-defined, extensible interfaces.

This tier provides the mathematical abstractions and is designed for flexibility. Components work with bare Flax modules and can be combined to create custom VMC engines.

If you want to work with exotic systems you might need to implement some custom objects from this tier, such as custom operators if you are working in Nuclear Physics. 
This is in general not a trivial task, but it's well supported and if you do everything else from NetKet will work fine with your custom objects.


```{list-table} Tier 1 Modules  
:header-rows: 1
:widths: 30 70

* - **Module**
  - **Description**
* - {mod}`netket.hilbert`
  - Hilbert space definitions and computational basis enumeration
* - {mod}`netket.operator`
  - Quantum operators, Hamiltonians, and expectation value computation
* - {mod}`netket.sampler`
  - Monte Carlo samplers and Markov chain transition rules
* - {mod}`netket.models`
  - Neural network architectures for quantum states
* - {mod}`netket.nn`
  - Neural network layers and building blocks
```

---

## Tier 2: High-Level Workflows

Opinionated, user-friendly abstractions that hide complexity and provide a streamlined interface for common tasks.

This tier is designed for ease of use and provides "batteries included" functionality for standard NQS optimization workflows.

It might be hard to fit your original ideas and custom training algorithms within the boundaries of this interface, so for algorithm research we expect that you might have to re-implement the login in the drivers yourself.


```{list-table} Tier 2 Modules
:header-rows: 1
:widths: 30 70

* - **Module**
  - **Description**
* - {mod}`netket.vqs`
  - Variational state abstractions ({class}`~netket.vqs.MCState`, {class}`~netket.vqs.MCMixedState`)
* - {mod}`netket.driver`
  - Optimization drivers ({class}`~netket.driver.VMC`, {class}`~netket.driver.SteadyState`)
* - {mod}`netket.optimizer`
  - Advanced optimizers with QGT and Stochastic Reconfiguration
* - {mod}`netket.logging`
  - Data logging and serialization utilities
* - {mod}`netket.callbacks`
  - Optimization callbacks for monitoring and control
```


- **Integrated Workflows**: Complete optimization loops with logging and checkpointing
- **Advanced Optimizers**: Built-in Quantum Geometric Tensor and Stochastic Reconfiguration
- **Automatic Management**: Handles sampling, parameter updates, and state management
- **Rich Logging**: Comprehensive data collection and analysis tools

---

## How to approach NetKet:

**Beginners:**: Use Tier-2 functionality from {class}`~netket.vqs.MCState`, {class}`~netket.driver.VMC`, and pre-built models from {mod}`netket.models`. This provides a complete, working NQS simulation with minimal complexity.

**Algorithm development:**: If you want to invent new optimization protocols or algorithms, you should re-implement some of the Tier-2 functionality yourself. We suggest to start by copy-pasting a driver such as the {class}`~netkt.driver.VMC_SR` and editing it directly, such that you will still benefit from the loggers and more..

**Simulating systems not built-in:**: If you want to work on composite systems, or nuclear physics models, or complicated continuous space setups, you will have to implement some custom operators (Tier-1). You will not need, in principle, to re-implement other parts of NetKet.


## Editing NetKet internals directly

NetKet has been designed to allow users to customize most things without the need to edit NetKet's files directly.
While editing the source code of the installed NetKet version directly might seem a simple and fast way to achieve what you want, it has the downside that you will not be able to (i) easily update to new NetKet versions and (ii) it is very hard to share your code with others.

Instead, if you want to change something, you should be able to copy the relevant files, and then import them from anywhere else.