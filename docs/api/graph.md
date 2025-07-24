(netket_graph_api)=
# netket.graph

```{eval-rst}
.. currentmodule:: netket.graph

```

This module provides graphs on which local Hamiltonians can be defined.

```{eval-rst}
.. inheritance-diagram:: netket.graph
   :top-classes: netket.graph.AbstractGraph
   :parts: 1

```

## Abstract Classes

Below you find a list of all public classes defined in this module.

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class
   :nosignatures:

   AbstractGraph

```

## Concrete Classes

Below you find a list of all concrete classes that you can use.

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class
   :nosignatures:

   Graph
   Lattice
   Edgeless
   lattice.LatticeSite

```

## Pre-built Lattices

### Simple hypercubic lattices

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class
   :nosignatures:

   Grid
   Hypercube
   Square
   Cube

```

### Other lattices

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class
   :nosignatures:

   Chain
   Triangular
   Honeycomb
   KitaevHoneycomb
   Kagome
   BCC
   FCC
   Diamond
   Pyrochlore

```

## Handling lattice symmetries

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class
   :nosignatures:

   space_group.SpaceGroup
   space_group.TranslationGroup

```
## Additional functions

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class

   DoubledGraph
   disjoint_union

```