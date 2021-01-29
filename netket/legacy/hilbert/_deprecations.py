from netket.utils import warn_deprecation

# To be removed in v3.1
def graph_to_N_depwarn(N, graph):

    if graph is not None:
        warn_deprecation(
            r"""
            The ``graph`` argument for hilbert spaces has been deprecated in v3.0.
			It has been replaced by the argument ``N`` accepting an integer, with 
			the number of nodese in the graph. 

			You can update your code by passing `N=_your_graph.n_nodes`.
			If you are also using Ising, Heisemberg, BoseHubbard or GraphOperator
			hamiltonians you must now provide them with the extra argument
			``graph=_your_graph``, as they no longer take it from the hilbert space.
			"""
        )

        if N == 1:
            return graph.n_nodes
        else:
            raise ValueError(
                "Graph object can only take one argumnent among N and graph (deprecated)."
            )

    return N
