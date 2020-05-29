import abc


class AbstractVariationalState(abc.ABC):
    """Abstract class for variational states representing either pure states
       or mixed quantum states.
       A variational state is a quantum state depending on a set of
       parameters, and that supports operations such
       as computing quantum expectation values and their gradients."""

    @abc.abstractmethod
    def mean(self, op):
        r"""Estimates the quantum expectation value for a given operator O.
            In the case of a pure state Psi, this is <O>= <Psi|O|Psi>/<Psi|Psi>
            otherwise for a mixed state Rho, this is <O> = Tr rho O / Tr rho.

        Args:
            op (netket.operator.AbstractOperator): the operator O.

        Returns:
            netket.stats.Stats: An estimation of the quantum expectation value <O>.
        """
        return self.mean_and_grad(op)[0]

    @abc.abstractmethod
    def grad(self, op):
        r"""Estimates the gradient of the quantum expectation value of a given operator O.

        Args:
            op (netket.operator.AbstractOperator): the operator O.

        Returns:
            array: An estimation of the average gradient of the quantum expectation value <O>.
        """
        return self.mean_and_grad(op)[1]

    @abc.abstractmethod
    def mean_and_grad(self, op):
        r"""Estimates both the gradient of the quantum expectation value of a given operator O.

        Args:
            op (netket.operator.AbstractOperator): the operator O

        Returns:
            netket.stats.Stats: An estimation of the quantum expectation value <O>.
            array: An estimation of the average gradient of the quantum expectation value <O>.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def samples(self, n_samples):
        r"""Returns samples of this quantum state from the computational basis.
            In the case of a pure state, samples x=(x_1,..,x_N) are returned according to the
            Born probability P(x)=|Psi(x)|^2. In the case of mixed states, P(x)=Rho(x,x).

        Args:
            n_samples (int): the number of samples required.

        Returns:
            array: An array of shape (n_samples,N) containing samples x.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameters(self):
        r"""Parameters of this variational state"""
        raise NotImplementedError

    @abc.abstractmethod
    def log_amplitude(self, x, xc=None, normalized=False):
        r"""Returns the logarithm of the amplitudes of the given variational state.
            In the case of a pure state, this is log(Psi(x)) for quantum numbers x in the computational basis.
            In the case of a mixed state, this is log(Rho(x,y)).

        Args:
            x (array): Batch of of quantum numbers of shape (batch_size,N) for pure states.
                       For mixed states, x has shape (batch_size,2N) unless xc is not None.
            y (array): If given, in the case of a mixed state it is used to compute the amplitude rho(x,y).
                        If None, it is instead assumed that x is the concatenation (x,y).
                        xc is ignored in the case of pure states.
            normalized (bool): Whether the normalized wave function log amplitudes are required.
                               Notice that for most variational states it is
                               exponentially expensive in N to compute the normalization.

            Returns:
                array: An array of size batch_size containing the log amplitudes.
        """
        raise NotImplementedError

    def grad_log_amplitude(self, x, xc=None, normalized=False):
        r"""Returns the gradient of the logarithm of the amplitudes of the given variational state
            with respect to the variational parameters.
            In the case of a pure state, this is grad(log(Psi(x))) for quantum numbers x in the computational basis.
            In the case of a mixed state, this is grad(log(Rho(x,y))).

        Args:
            x (array): Batch of of quantum numbers of shape (batch_size,N) for pure states.
                       For mixed states, x has shape (batch_size,2N) unless xc is not None.
            y (array): If given, in the case of a mixed state it is used to compute the amplitude rho(x,y).
                        If None, it is instead assumed that x is the concatenation (x,y).
                        xc is ignored in the case of pure states.
            normalized (bool): Whether the normalized wave function log amplitudes are required.
                               Notice that for most variational states it is
                               exponentially expensive in N to compute the normalization.

            Returns:
                array: An array of shape (batch_size,n_par) containing the gradient of the log amplitudes.
        """
        raise NotImplementedError

    @property
    def hilbert(self, number):
        r"""netket.hilbert.AbstractHilbert: The descriptor of the Hilbert space
                                            on which this variational state is defined.
        """
        raise NotImplementedError
