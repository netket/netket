#!/usr/bin/env python
# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Base class for NetKet input driver Graph objects.

'''

from pynetket.python_utils import set_opt_pars

# Test to see if networkx is installed
try:
    import networkx as nx
    import_nx = True
except ImportError:
    import_nx = False

# Test to see if igraph is installed
try:
    import igraph as ig
    import_ig = True
except ImportError:
    import_ig = False


class Graph(object):
    '''
    Driver for input graph parameters.

    Simple Usage::

        >>> graph = Graph("Hypercube", L=20, Dimension=1, Pbc=True)
        >>> print(graph._pars)
        {'Name': 'Hypercube', 'L': 20, 'Dimension': 1, 'Pbc': True}
    '''

    _name = "Graph"

    def __init__(self, name, **kwargs):
        '''
        Store the appropriate parameters to write to json input.

        Arguments
        ---------

        name : string
            Hypercube or Custom.


        kwargs
        ------

        L : int
            Used with Hypercube graphs. Sets the length per dimension. Default
            value is 10.

        Dimension : int
            Used with Hypercube graphs. Sets the number of dimensions. Default
            value is 1.

        Pbc : bool
            Used with Hypercube graphs. Determines periodicity. Default value is
            True.

        graph : nx.graph or igraph.graph
            Used with Custom graphs. Graph object that is used to populate the
            edges and edge colors (if applicable) values in our input
            parameters. The graph must have edges.
        '''

        self._pars = {}

        # Hypercube option
        if name == "Hypercube":

            self._pars['Name'] = name

            set_opt_pars(self._pars, "L", kwargs)
            set_opt_pars(self._pars, "Dimension", kwargs)
            set_opt_pars(self._pars, "Pbc", kwargs)

        elif name == "Custom":
            if "graph" in kwargs:
                if import_nx:
                    if type(kwargs["graph"]) == type(nx.Graph()):
                        # Grab edges
                        print("Found a networkx graph")
                        if len(kwargs["graph"].edges) <= 0:
                            raise AssertionError(
                                "Graph doesn't have more than 0 edges.")
                        self._pars['Edges'] = list(kwargs["graph"].edges)

                        # Grab edge colors
                        try:
                            self._pars['EdgeColors'] = [[
                                u, v, kwargs["graph"][u][v]['color']
                            ] for u, v in kwargs["graph"].edges]

                        except KeyError:
                            print("No edge colors found.")

                # TODO add import igraph

        else:
            raise ValueError("%s graph not supported" % name)


if __name__ == '__main__':
    graph = Graph("Hypercube", L=20, Dimension=1, Pbc=True)
    print(graph._pars)
