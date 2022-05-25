(QGT_and_sr)=

# Quantum Geometric Tensor and Stochastic Reconfiguration

The Quantum Geometric Tensor (QGT) is the Fubini-Study metric tensor of the manifold on which a variational state is defined.
In the general case (and in practise for any NQS ansatz), the QGT varies depending on the current quantum state and thus needs to be computed when the variational parameters $W$ change.
To give an example, we consider a variational ansatz $ \psi\colon \mathbb{R} \rightarrow \mathscr{H} $ which maps elements $W$ of the parameter space to vectors in Hilbert space (quantum states) $\psi_W$.

The parameter space has a simple Euclidean metric, therefore the distance between two points $\bf{W}$ and $\bf{W} + \bf{\delta W}$ is simply  $\Vert\bf{\delta W} \Vert$.
However, what we really are interested in when optimizing variational ansätze is not the (Euclidean) distance between those two points in parameter space, but rather the distance in the Hilbert space between the corresponding quantum states (which also properly takes into account gauge degrees of freedom).
The quantum mechanical distance function for quantum states is the Fubini-Study distance $ d(\psi, \phi) = \cos^{-1} \sqrt{\frac{\langle\psi|\phi\rangle \langle\phi|\psi\rangle}{\langle\psi|\psi\rangle \langle\phi|\phi\rangle}} $.
This can be expanded to second order in an infinitesimal parameter change $\delta W$ as $ d(\psi_W, \psi_{W + \delta W}) = (\delta W)^\dagger S \delta W $ where $ S $ is the QGT.
In NetKet you can obtain (an approximation of) the quantum geometric tensor of a variational state by calling {attr}`~netket.vqs.VariationalState.quantum_geometric_tensor`.

```python
ma = nk.models.RBM(alpha=1, dtype=float)
sa = nk.sampler.MetropolisLocal(nk.hilbert.Spin(0.5, 16), n_chains=16)
vs = nk.vqs.MCState(sa, ma)

qgt = vs.quantum_geometric_tensor()
```

This will return an object that behaves as a Matrix, which acts on PyTrees of parameters.

```python
_, grad = vs.expect_and_grad(nk.operator.spin.sigmax(nk.hilbert.Spin(0.5, 16), 0))
qgt_times_grad = qgt@grad
```

The quantum geometric tensor also acts on dense ravellings of the parameters

```python
grad_dense, unravel = nk.jax.tree_ravel(grad)
qgt_times_grad_dense = qgt@grad_dense
```

You can convert the quantum geometric tensor to a matrix representation by calling `qgt.to_dense()`.

```python
qgt_dense = qgt.to_dense()
```

Lastly, you can solve the linear system $ Q_{i,j} x_j = F_i $ by calling the solve method:

```python
x, info = qgt.solve(jax.scipy.sparse.linalg.gmres, grad)
x, info = qgt.solve(nk.optimizer.solver.cholesky, grad)
```

While mathematically those operations are all well defined, there are several ways to implement them in code, all with different performance characteristics. For that reason, we have several (3) different implementations of the same Quantum Geometric Tensor object.
The 3 implemnentations are:

 - {ref}`netket.optimizer.qgt.QGTOnTheFly`, which uses jax automatic differentiation through two `vjp` and one `jvp` product to compute the action of quantum geometric tensor on a vector and operates natively on PyTrees. This method will essentially run AD every time you compute `QGT@vector`. This method shines if the parameters of your network are stored in a PyTree with few leaf nodes and/or you are not performing many iterations of the iterative solver. It can compute the full dense QGT but it is not efficient at doing it so we advise not to use it with dense solvers.
 - {ref}`netket.optimizer.qgt.QGTJacobianDense`, which precomputes the log derivatives ( $ O_k $ ) when it's constructed and converts it to a single dense array. If you have a high number of total parameters and/or many leaf nodes in your parameter PyTree, this implementation might perform better because everything is stored contiguously in memory. However, it has an high 'startup cost'. 
 - {ref}`netket.optimizer.qgt.QGTJacobianPyTree`, same as above, but the precomputed jacobian is not stored contiguously in memory but is stored as a PyTree. This might work better than `QGTJacobianDense` if there are few leaf nodes. We haven't studied the performance tradeoffs between the two Jacobian implementations and we would appreciate feedback.

We also have an extra implementation, called `netket.optimizer.qgt.QGTAuto`, which uses some heuristics based on the parameters of the network to select the best QGT implementation. Be warned that the heuristics we use is very crude, and might not pick the best implementation all the time.

All the QGT implementations listed above have several options that can affect their performance. 
We advise you to have a look at them and experiment.
We provide a lot of freedom because it's not yet clear to us what is the best implementation for which kind of problems.
If you work with NetKet and determine that a particular implementation works best for certain types of networks, we would be glad to hear it! Let us know by opening a Discussion on our GitHub repository. We might use the insight to improve the automatic selection.


## Stochastic Reconfiguration

The [stochastic reconfiguration (SR) method](https://www.attaccalite.com/PhDThesis/html/node15.html) is a technique that makes use of the information encoded in the quantum geometric tensor described above to _precondition_ the gradient used in stochastic optimization, _improving_ the convergence rate to the ground state of a Hamiltonian $ \hat{H} $.

We would like to underline that SR can be derived (and therefore thought of) as imaginary time evolution of the variational ansatz.
The derivation in the case of a variational optimization for the ground state, can be sketched as follows:

Given a a variational wavefunction $ \ket{\psi_W} $, we consider its first order Taylor expansion around $ W $, $ \ket{\psi_{W+\delta W}} = \ket{\psi_W} + \delta W_k \hat{O}_k \ket{\psi_W} $, where $ \bra{\sigma}\hat{O}_k \ket{\eta} = \delta_{\sigma,\eta} \frac{d \log\psi_W(\sigma)}{dW_k}$.
We wish to determine the updates $ \delta_{\sigma,\eta} $ of the variational parameters that match a step of imaginary-time evolution, given by 

\begin{equation}
\ket{\phi} = U(\epsilon)\ket{\psi_W} = e^{\epsilon\hat{H}}\ket{\psi_W} \sim (\mathbb{I} - \epsilon \hat{H})\ket{\psi_W}
\end{equation}

It is possible to show that the updates $\delta W_k $ that minimise the norm of the state $\ket{\phi}-\ket{\psi_{W+\delta W}}$ can be determined by solving the linear system 

\begin{equation}
S_{i,k} \delta W_k = F_i
\end{equation}

where $ F_i = \langle \hat{E}^{loc} \hat{O}_i\rangle - \langle \hat{E}^{loc} \rangle\langle \hat{O}_i\rangle $ is the gradient of the Energy and $ S_{i,k} = \langle \hat{O}^\dagger_i \hat{O}_k\rangle - \langle \hat{O}^\dagger_i \rangle\langle \hat{O}_k\rangle $ is the Quantum Geometric Tensor.
The QGT is positive definite, therefore it can be inverted and the solution is formally written as

\begin{equation}
\bf{\delta W} = S^{-1} \bf{F},
\end{equation}
where bold fonts are used for vectors.
A complication is given by the fact that the QGT is determined by Monte Carlo sampling and it might have several eigenvalues that are zero or very small, leading to numerical stability issues when inverting the matrix or in the resulting dynamics.
The linear system can be solved with several methods. For the models with many parameters and to achieve the best performance, iterative solvers such as those found in [jax.scipy.sparse.linalg](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.sparse.linalg), such as {func}`jax.scipy.sparse.linalg.cg` {func}`jax.scipy.sparse.linalg.gmres` are the best choice. 
Do note that to stabilize those algorithms it is often needed to add a small ($10^{-5} - 10^{-2}$) shift to the diagonal of the QGT. 
This can be set with the keyword argument `diag_shift`.
Those methods, combined with our lazy representations of the QGT, never instantiate the full matrix and therefore achieve a great performance.
However, when the number of parameters is small (smaller than 1000-5000), it might make sense to solve the system by factorizing the QGT with cholesky or SVD.
Those techniques require instantiating the full dense matrix, but in general are more stable and don't require a diagonal shift.

### Using stochastic reconfiguration

To use SR, you must simply provide it as a preconditioner to the VMC solver.

```python
sr = nk.optimizer.SR()
gs = nk.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)
```

By default this will use an appropriate QGT and the iterative solver `jax.scipy.sparse.linalg.cg`.
It is possible to change the iterative solver by providing any solver from `jax.scipy.sparse.linalg` or one of the dense solvers in `netket.optimizer.solver` (such as svd/cholesky/LU) to the SR object. 

```python
sr = nk.optimizer.SR(solver=nk.optimizer.solver.cholesky)
gs = nk.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)
```

The solver should be a function that accepts two arguments: the S matrix and the F vector in the linear system to be solved, and an optional x0 keyword argument.
If you want to specify options of a linear solver, such as the tolerance or the cutoff rate you can do the following:

```python
from functools import partial
sr = nk.optimizer.SR(solver=partial(jax.scipy.sparse.linalg.gmres, maxiter=1000, tol=1e-8))
gs = nk.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)

```
If you don't specify the QGT format, NetKet will try to guess the best format.
We reccomend you experiment and specify the QGT format that gives you the best performance, which can be by passing it as an argument. Additional keyword arguments will be forwarded to the QGT constructor, as shown below:

```python
sr = nk.optimizer.SR(QGTJacobianPyTree, solver=partial(jax.scipy.sparse.linalg.gmres, maxiter=1000, tol=1e-8)diag_shift=1e-3
gs = nk.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)
```

Since SR leads to an optimisation that approximates an imaginary time evolution, we find that in general it is not a good idea to couple SR with advanced optimisers like ADAM, which modify the gradient remarkably. Stochastic Gradient Descent is the best choice in general.
