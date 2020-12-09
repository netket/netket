from .custom_hilbert import CustomHilbert


class Qubit(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local qubit states."""

    def __init__(self, N=1):
        r"""Initializes a qubit hilbert space.

        Args:
        graph: Graph representation of qubits. If None, size
              is used to fix the total number of qubits.
        size: Number of qubits. If None, a graph must be speficied.


        Examples:
            Simple spin hilbert space.

            >>> from netket.graph import Hypercube
            >>> from netket.hilbert import Qubit
            >>> g = Hypercube(length=10,n_dim=2,pbc=True)
            >>> hi = Qubit(graph=g)
            >>> print(hi.size)
            100
        """
        super().__init__([0, 1], N)

    def __repr__(self):
        return "Qubit(N={})".format(self._size)
