# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NetKet is a Python library for studying many-body quantum systems using neural networks and machine learning techniques. Built on JAX, it provides tools for variational quantum simulations, quantum Monte Carlo methods, and neural quantum states.

## Development Setup

NetKet development uses **uv** as the preferred package manager. Install uv from https://docs.astral.sh/uv/getting-started/installation/

### Environment Setup
- Install NetKet in development mode: `uv sync`
- Activate environment: `uv shell` or prefix commands with `uv run`

### Testing
- Run all tests: `uv run pytest`
- Run specific test file: `uv run pytest test/path/to/test_file.py`
- Run tests with coverage: `uv run pytest --cov=netket`
- Run tests in parallel: `uv run pytest -n auto`

### Distributed Computing Tests
NetKet has built-in distributed computing support via JAX sharding. Test distributed functionality:

**CPU Sharding Mode (Local Testing):**
- Run tests with multiple CPU devices: `NETKET_EXPERIMENTAL_SHARDING_CPU=2 uv run pytest`
- This simulates distributed computing on a single machine with 2 virtual processes

**Multi-process Distributed Testing:**
- Test true multi-process distributed: `uv run djaxrun -np 2 pytest`
- Uses actual separate processes for distributed testing
- The `djaxrun` command is NetKet's distributed JAX runner

**Important:** Always test changes in both standard and distributed modes, as distributed computing can expose different bugs and behavior patterns.

### Code Quality
- Format code: `uv run black .`
- Lint code: `uv run ruff check .`
- Fix linting issues: `uv run ruff check --fix .`
- Use pre-commit for commits: `uv run pre-commit run --all-files --verbose`

### Documentation
- Build docs: `uv run make -C docs html`
- View docs locally: Open `docs/_build/html/index.html`

### Package Management
- Install development dependencies: `uv sync` (preferred)
- Build package: `uv run python -m build`

## NetKet Architecture

NetKet follows a **three-tier modular architecture** designed to serve different user needs and development scenarios. Understanding this architecture is crucial for effective development and extension.

### Three-Tier Architecture

**Tier 0: Core JAX Tools**
- **Purpose**: Low-level JAX utilities for performance and distributed computing
- **Users**: Algorithm developers, performance optimization
- **Modules**:
  - `netket.jax`: Chunking (`vmap_chunked`, `vjp_chunked`), distributed computing, custom JAX operations
  - `netket.utils.struct`: PyTree implementations, immutable data structures
  - `netket.utils.timing`: Performance profiling tools
  - `netket.utils.numbers`: Numerical utilities and special types

**Tier 1: VMC Building Blocks**
- **Purpose**: Extensible components for building variational algorithms
- **Users**: Algorithm researchers, custom system implementations
- **Modules**:
  - `netket.hilbert`: Hilbert space definitions (`Spin`, `Fock`, `Qubit`, `DoubledHilbert`)
  - `netket.operator`: Quantum operators (`Ising`, `Heisenberg`, `LocalOperator`, `PauliStrings`)
  - `netket.sampler`: Monte Carlo samplers (`MetropolisLocal`, `MetropolisExchange`, `ParallelTempering`)
  - `netket.models`: Neural network architectures (`RBM`, `GCNN`, `ARNN`, `Jastrow`)
  - `netket.nn`: Neural network layers and building blocks
  - `netket.graph`: Graph structures (`Lattice`, space group symmetries)

**Tier 2: High-Level Workflows**
- **Purpose**: Opinionated, easy-to-use abstractions for standard workflows
- **Users**: General users, beginners, standard optimization tasks
- **Modules**:
  - `netket.vqs`: Variational states (`MCState`, `MCMixedState`, `FullSumState`)
  - `netket.driver`: Optimization drivers (`VMC`, `SteadyState`)
  - `netket.optimizer`: Advanced optimizers with QGT and Stochastic Reconfiguration
  - `netket.logging`: Data logging and serialization
  - `netket.callbacks`: Optimization callbacks and monitoring

### Key Design Patterns

**Multiple Dispatch**: NetKet extensively uses `@dispatch` decorators for flexible method definitions:
```python
@expect.dispatch
def expect(vstate: MCState, operator: LocalOperator):
    # Implementation specific to MCState + LocalOperator
    
@expect.dispatch  
def expect(vstate: FullSumState, operator: AbstractOperator):
    # Different implementation for exact computation
```

**PyTree-Based Immutability**: All NetKet objects are immutable PyTrees:
- Parameters are frozen dictionaries that must be copied to modify
- States are immutable and hashable for use in JAX transformations
- All objects can be serialized/deserialized with Flax

**JAX Integration**: All computations are JAX-native:
- Custom JAX operations in `netket.jax` (e.g., complex-aware `vjp`)
- Automatic chunking for memory efficiency (`vmap_chunked`, `vjp_chunked`)
- Built-in distributed computing via JAX sharding (enabled by default)
- JIT compilation support throughout

**Extensible Interfaces**: Tier 1 components have well-defined extension points:
- Custom operators: Implement `AbstractOperator` and define `expect` methods via dispatch
- Custom samplers: Inherit from base sampler classes and implement transition rules
- Custom Hilbert spaces: Extend `AbstractHilbert` or `DiscreteHilbert`

### Development Guidelines by Tier

**Working with Tier 0**:
- Use when implementing high-performance custom algorithms
- Essential for distributed computing and memory optimization
- Example: Use `vmap_chunked` for large-scale expectation value computations

**Extending Tier 1**:
- Implement custom operators for exotic systems (nuclear physics, continuous space)
- Create custom samplers for specialized transition rules  
- Define custom Hilbert spaces with constraints
- All other NetKet functionality will work with your custom components

**Using Tier 2**:
- Start here for standard NQS workflows
- Use `MCState` + `VMC` driver for most ground state problems
- Leverage built-in optimizers like `SR` (Stochastic Reconfiguration)
- For custom training loops, copy and modify existing drivers

### Distributed Computing

NetKet has built-in distributed computing support:
- **Automatic**: JAX sharding is enabled by default, uses all visible GPUs
- **Testing**: Use `djaxrun -np N python script.py` to test multi-process locally
- **Multi-node**: Add `jax.distributed.initialize()` at script start for cluster usage
- **Performance**: See `docs/distributed-computing.md` for detailed guidance

## Testing Framework

Tests use pytest with custom fixtures for MPI, distributed JAX, and device detection. Test configuration handles:
- JAX cache clearing to prevent OOM
- MPI and distributed testing scenarios
- Device-specific test adaptation
- Autoregressive network test rate limiting

## Performance Considerations

### JAX Optimization
- Use `jax.jit` for performance-critical functions
- Leverage vectorization with `jax.vmap` 
- Consider chunking for large systems using `netket.jax` utilities (`vmap_chunked`, `vjp_chunked`)
- Use appropriate precision (float32 vs float64) based on requirements

### Memory Management
- Use chunking to avoid OOM errors: `MCState(chunk_size=128)` or similar
- NetKet automatically handles memory-efficient operations via Tier 0 utilities
- For custom algorithms, use `netket.jax.apply_chunked` and related functions

### Distributed Computing
- Distributed computing is enabled by default (uses all visible GPUs)
- Control GPU visibility: `CUDA_VISIBLE_DEVICES=0,1` before launching Python
- For multi-node: Add `jax.distributed.initialize()` at script start
- Test locally with `djaxrun` before deploying to clusters

### Development Performance Tips
- When extending Tier 1 components, ensure JAX compatibility for JIT compilation
- Use `netket.utils.struct.Pytree` base class for custom objects
- Implement multiple dispatch methods for optimal performance per component combination
- Profile using `netket.utils.timing` tools when optimizing custom algorithms