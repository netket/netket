# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from netket.utils import warn_deprecation


# To be removed in v3.1
def graph_to_N_depwarn(N, graph):

    if graph is not None:
        warn_deprecation(
            r"""
            The ``graph`` argument for hilbert spaces has been deprecated in v3.0.
            It has been replaced by the argument ``N`` accepting an integer, with
            the number of nodes in the graph.

            You can update your code by passing `N=_your_graph.n_nodes`.
            If you are also using `Ising`, `Heisenberg`, `BoseHubbard` or `GraphOperator`
            Hamiltonians you must now provide them with the extra argument
            ``graph=_your_graph``, as they no longer take it from the Hilbert space.
            """
        )

        if N == 1:
            return graph.n_nodes
        else:
            raise ValueError(
                "Graph object can only take one argument among N and graph"
                "(deprecated)."
            )

    return N
