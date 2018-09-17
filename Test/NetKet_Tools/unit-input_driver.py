#!/usr/bin/env python
'''
 Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import unittest
import os
import json
import numpy as np
import netket_tools as nkt

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


class KnownOutput(unittest.TestCase):
    def test1_nx_graph(self):
        if import_nx:
            L = 20
            G = nx.Graph()
            for i in range(L):
                G.add_edge(i, (i + 1) % L, color=1)
                G.add_edge(i, (i + 2) % L, color=2)

            edge_colors = [[u, v, G[u][v]['color']] for u, v in G.edges]

            # Specify custom graph
            pars = {}
            pars['Graph'] = {
                'Edges': list(G.edges),
                'EdgeColors': edge_colors,
            }

            self.assertEqual(pars["Graph"], nkt.graph.from_networkx(G))

    def test2_ig_graph(self):
        if import_ig:
            L = 20
            G = nx.Graph()
            for i in range(L):
                G.add_edge(i, (i + 1) % L, color=1)
                G.add_edge(i, (i + 2) % L, color=2)

            edge_colors = [[u, v, G[u][v]['color']] for u, v in G.edges]

            # Specify custom graph
            pars = {}
            pars['Graph'] = {
                'Edges': list(G.edges),
                'EdgeColors': edge_colors,
            }

            self.assertEqual(pars["Graph"], nkt.graph.from_networkx(G))


if __name__ == "__main__":
    print("Testing Python NetKet Tools")
    unittest.main()
