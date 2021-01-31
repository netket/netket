
.. _graph:

###############
Graph submodule
###############

.. currentmodule:: netket.graph

.. _graph-stuf:

Stuff
-----

This document describes the :ref:`graph-api` submodule. 

Most objects in Netket, such as Neural Quantum States, Operators
and Hilbert spaces are defined on top of a Graph, which describes the
physical lattice on top of which said object is defined.

The abstract interface is defined within [AbstractGraph](#netket.graph.AbstractGraph), 
and a standard implementation is based upon the `NetworkX <https://networkx.org/>`_ library.

Given a list of edges and vertices, it is possible to construct the NetworkX graph with the following
function:

.. autofunction:: Graph
 
 
However, for ease of use, the [Graph submodule](#netket.graph) provides several constructors for
common lattices, such as N-dimensional grids, HyperCubes, Graphs without edges and several others.

The full list is located in:

.. autosummary::
   :toctree: _generated/graph
   :nosignatures:

   netket.graph.Edgeless
   netket.graph.Hypercube
   netket.graph.Square
   netket.graph.Grid
   netket.graph.Lattice

.. _graph-abstract-interface:

Abstract Interface
------------------

All graphs inherit from the [AbstractGraph](#netket.graph.AbstractGraph) base class, and you should check
it's documentation to see the commplete list of supported properties.

In general, with any graph you can do manipulations like the following:

.. code-block:: python

  import netket as nk

  nk.graph.Square(4)



.. autoclass:: netket.graph.AbstractGraph
  :show-inheritance:




.. autoclass:: netket.graph.NetworkX
     :show-inheritance:


Lattice Constructors
--------------------

.. autoclass:: Grid
    :no-members: 
    :no-inherited-members:
 

.. autofunction:: 
  Graph
  Grid        
  Chain
  Edgeless
  Lattice
  Hypercube
  Square
