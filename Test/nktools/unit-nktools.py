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
import nktools as nkt

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
    @unittest.skipIf((not import_nx), "No NetworkX module found.")
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
                'IsBipartite': False,
            }

            self.assertEqual(pars["Graph"], nkt.graph(G))

    @unittest.skipIf((not import_nx), "No NetworkX module found.")
    def test1_nx_graph(self):
        if import_nx:
            L = 20
            G = nx.Graph()
            for i in range(L):
                G.add_edge(i, (i + 1) % L)

            # Specify custom graph
            pars = {}
            pars['Graph'] = {
            'Edges': list(G.edges),
            'IsBipartite': True,
            }

            self.assertEqual(pars["Graph"], nkt.graph(G))

    @unittest.skipIf((not import_ig), "No iGraph module found.")
    def test2_ig_graph(self):
        print(import_ig)
        print(not import_ig)

        if import_ig:
            L = 20
            G = ig.Graph([(x, (x + 1) % L) for x in range(L)])
            G.add_edges([(x, (x + 2) % L) for x in range(L)])

            G.es['color'] = [1, 2] * L

            edge_colors = [[u, v, G.es['color']] for u, v in G.get_edgelist()]

            # Specify custom graph
            pars = {}
            pars['Graph'] = {
                'Edges': list(G.get_edgelist()),
                'EdgeColors': edge_colors,
                'IsBipartite': False,
            }

            self.assertEqual(pars["Graph"], nkt.graph(G,automorphisms=False))


if __name__ == "__main__":
    print("Testing Python NetKet Tools")
    unittest.main()
