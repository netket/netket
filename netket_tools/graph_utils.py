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
"""
Input script helper functions for creating custom graphs.
"""

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


def graph(graph):
    """
    Takes a networkx (or igraph) graph as input and extracts necessary data to
    write to NetKet Graph parameters.

    Arguments
    ---------

        graph : (nx.Graph or ig.igraphs)
             Custom graph.

    Returns
    -------

        pars : (dict)
            Dictionary of parameters to be dumped to json input file.
    """

    # Parameters (that will be dumped to json input file)
    pars = {}

    # Check for networkx graph
    if import_nx:
        if isinstance(graph, type(nx.Graph())):

            # Extract edge data
            print("Found a networkx graph")
            if len(graph.edges) <= 0:
                raise AssertionError("Graph has fewer than one edge.")

            pars['Edges'] = list(graph.edges)

            # Grab edge colors
            try:
                pars['EdgeColors'] = [[u, v, graph[u][v]['color']]
                                      for u, v in graph.edges]
            except KeyError:
                print("No edge colors found.")

            return pars

    # Check for igraph graph
    elif import_ig:
        if isinstance(graph, type(ig.Graph())):

            # Extract edge data
            print("Found an igraph graph")
            if len(graph.get_edgelist()) <= 0:
                raise AssertionError("Graph has fewer than one edge.")
            pars['Edges'] = list(graph.get_edgelist())

            # Grab edge colors
            try:
                pars['EdgeColors'] = [[u, v, graph.es['color']]
                                      for u, v in graph.get_edgelist()]

            except KeyError:
                print("No edge colors found.")

            return pars

    else:
        print("Neither igraph nor networkx could be imported")
        raise
