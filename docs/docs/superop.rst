============================
The Lindblad Master Equation
============================

In short: Many-body states in NetKet are stored in `big-endian format <https://en.wikipedia.org/wiki/Endianness#Bi-endianness>`_, and that applies to super-operators too.

This means that, if the local basis of your Hilbert space is :code:`[0,1]`, the joint Hilbert space
composed of two sites will have 4 states, indexed in the following order: :code:`[00,01,10,11]`. 
This format is called Big-Endian because increasing the first element in the array will increase the 
index within the hilbert space more than increasing the last element in the array.

This format is also the most natural in row-column languages such as Python/Numpy, as opposed to Julia 
and Fortran which are column-major.
It is the format that arises if you take the kronecker product with numpy.

This same ordering is then used when representing operators, as they inherit the ordering of the hilbert
space along both of their dimensions.

The Density Matrix and the Liouvillian
--------------------------------------

The lindblad master equation that is encoded in the liouvillian is:

.. math ::

    \mathcal{L} = -i \left[ \hat{H}, \hat{\rho}\right] + \sum_i \left[ \hat{L}_i\hat{\rho}\hat{L}_i^\dagger -
        \left\{ \hat{L}_i^\dagger\hat{L}_i, \hat{\rho} \right\} \right]

The liouvillian is a rank-4 tensor obtained by taking the kronecker product of 
two objects in the operator-space, and one can again choose row-stacking or column-stacking
to represent it.

Qutip and QuantumOptics.jl both use column-stacking, the first for historical reasons (due to 
their Matlab origins), the other to leverage the column-major nature of Julia. In NetKet 3 we 
switched to a row-major (row-stacking) encoding as it is more natural in Python and it simplifies
working with operators. 

For performance reasons we pack together the Unitary and anti-Unitary terms into a single
non hermitian Hamiltonian :math:`\hat{H}_{nh}`, which allows us to rewrite the formula above as

That is then composed with the jump operators in the inner kernel with the formula:

.. math ::

    \mathcal{L} = -i \hat{H}_{nh}\hat{\rho} +i\hat{\rho}\hat{H}_{nh}^\dagger + \sum_i \hat{L}_i\hat{\rho}\hat{L}_i^\dagger

Where :math:`\hat{H}_{nh}` is given by the formula:

.. math ::

    \hat{H}_{nh} = \hat{H} - \sum_i \frac{i}{2}\hat{L}_i^\dagger\hat{L}_i


The row-stacked, matrix-representation of the Liouvillian is then given by the following formula:

.. math ::
	\hat{\mathcal{L}} = -i \hat{H}_{nh} \otimes \hat{I} + i \hat{I} \otimes \hat{H}_{nh}^\star
	+ \sum_i  \hat{L}_i\otimes\hat{L}_i^\star

An intuitive derivation of the formula above can be had by trying to compute the connected elements of
:math:`\langle \sigma | \hat{\mathcal{L}}(\hat{\rho})|\eta\rangle` connecting to 
:math:`\langle x | \hat{\rho} | y \rangle`.

Alternatively, a formal derivation of the formula above is laid out by my fellow Fabrizio in the appendix 
A1 of `this article <https://arxiv.org/pdf/1909.11619.pdf#page=16>`_.



Computing Observables
---------------------

Expectation values of operators can be computed on a mixed state according to the well known formula

.. math ::
	\langle \hat{O} \rangle &= \mathrm{Tr}[\hat{O}\hat{\rho}]  \\
			&= \sum_{\sigma,\eta} \langle \sigma | \hat{O} |\eta\rangle \langle\eta |\hat{\rho} |\sigma \rangle \\
			& = \sum_{\sigma,\eta} \frac{\langle \sigma | \hat{\rho} | \sigma\rangle}{\langle \sigma | \hat{\rho} | \sigma\rangle}\langle \sigma | \hat{O} |\eta\rangle \langle\eta |\hat{\rho} |\sigma \rangle\\
			&= \sum_{\sigma} \langle \sigma | \hat{\rho} | \sigma\rangle \left(\sum_\eta \langle \sigma | \hat{O} |\eta\rangle \frac{\langle\eta |\hat{\rho} |\sigma \rangle}{\langle \sigma | \hat{\rho} | \sigma\rangle}\right)

And since the diagonal elements of the density matrix are positive, real and normalized to 1, they form a well defined probability distribution that we can sample and use to estimate observables.

The only downside of this approach is that the Markov-Chain that is needed to sample the diagonal elements
is not the same as the one that can be used to sample the whole density matrix, which is used during optimization of the steady-state.



