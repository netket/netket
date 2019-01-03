# GraphOperator
A custom graph based operator.
## Constructor
Constructs a new ``GraphOperator`` given a hilbert space and either a
list of operators acting on sites or a list acting on the bonds.
Users can specify the color of the bond that an operator acts on, if
desired. If none are specified, the bond operators act on all edges.

|    Field     |            Type            |                                                                               Description                                                                               |
|--------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|hilbert       |netket.hilbert.Hilbert      |Hilbert space the operator acts on.                                                                                                                                      |
|siteops       |List[List[List[complex]]]=[]|A list of operators that act on the nodes of the graph. The default is an empty list. Note that if no siteops are specified, the user must give a list of bond operators.|
|bondops       |List[List[List[complex]]]=[]|A list of operators that act on the edges of the graph. The default is an empty list. Note that if no bondops are specified, the user must give a list of site operators.|
|bondops_colors|List[int]=[]                |A list of edge colors, specifying the color each bond operator acts on. The defualt is an empty list.                                                                    |
### Examples
Constructs a ``BosGraphOperator`` operator for a 2D system.

```python
>>> from mpi4py import MPI
>>> import netket as nk
>>> sigmax = [[0, 1], [1, 0]]
>>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
>>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
>>> g = nk.graph.CustomGraph(edges=edges)
>>> hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], graph=g)
>>> op = nk.operator.GraphOperator(
    hi, siteops=[sigmax], bondops=[mszsz], bondops_colors=[0])
>>> ha.hilbert
<netket.hilbert.CustomHilbert object at 0x2b19b8298340>

```


## Properties
|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |const AbstractHilbert&| ``Hilbert`` space of operator.|

