# LocalOperator
A custom local operator.

## Class Constructor [1]
Constructs a new ``LocalOperator`` given a hilbert space and (if
specified) a constant level shift.

|Argument|              Type              |               Description               |
|--------|--------------------------------|-----------------------------------------|
|hilbert |netket._C_netket.hilbert.Hilbert|Hilbert space the operator acts on.      |
|constant|float=0.0                       |Level shift for operator. Default is 0.0.|

### Examples
Constructs a ``LocalOperator`` without any operators.

```python
>>> from netket.graph import CustomGraph
>>> from netket.hilbert import CustomHilbert
>>> from netket.operator import LocalOperator
>>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
>>> hi = CustomHilbert(local_states=[1, -1], graph=g)
>>> empty_hat = LocalOperator(hi)
>>> print(len(empty_hat.acting_on))
0

```


## Class Constructor [2]
Constructs a new ``LocalOperator`` given a hilbert space, a vector of
operators, a vector of sites, and (if specified) a constant level
shift.

|Argument |              Type              |                       Description                        |
|---------|--------------------------------|----------------------------------------------------------|
|hilbert  |netket._C_netket.hilbert.Hilbert|Hilbert space the operator acts on.                       |
|operators|List[List[List[complex]]]       |A list of operators, in matrix form.                      |
|acting_on|List[List[int]]                 |A list of sites, which the corresponding operators act on.|
|constant |float=0.0                       |Level shift for operator. Default is 0.0.                 |

### Examples
Constructs a ``LocalOperator`` from a list of operators acting on
a corresponding list of sites.

```python
>>> from netket.graph import CustomGraph
>>> from netket.hilbert import CustomHilbert
>>> from netket.operator import LocalOperator
>>> sx = [[0, 1], [1, 0]]
>>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
>>> hi = CustomHilbert(local_states=[1, -1], graph=g)
>>> sx_hat = LocalOperator(hi, [sx] * 3, [[0], [1], [5]])
>>> print(len(sx_hat.acting_on))
3

```


## Class Constructor [3]
Constructs a new ``LocalOperator`` given a hilbert space, an
operator, a site, and (if specified) a constant level
shift.

|Argument |              Type              |                       Description                        |
|---------|--------------------------------|----------------------------------------------------------|
|hilbert  |netket._C_netket.hilbert.Hilbert|Hilbert space the operator acts on.                       |
|operator |List[List[complex]]             |An operator, in matrix form.                              |
|acting_on|List[int]                       |A list of sites, which the corresponding operators act on.|
|constant |float=0.0                       |Level shift for operator. Default is 0.0.                 |

### Examples
Constructs a ``LocalOperator`` from a single operator acting on
a single site.

```python
>>> from netket.graph import CustomGraph
>>> from netket.hilbert import CustomHilbert
>>> from netket.operator import LocalOperator
>>> sx = [[0, 1], [1, 0]]
>>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
>>> hi = CustomHilbert(local_states=[1, -1], graph=g)
>>> sx_hat = LocalOperator(hi, sx, [0])
>>> print(len(sx_hat.acting_on))
1

```



## Class Methods 
### conjugate
Returns the complex conjugation of this operator


### get_conn
Member function finding the connected elements of the Operator. Starting
from a given visible state v, it finds all other visible states v' such
that the matrix element O(v,v') is different from zero. In general there
will be several different connected visible units satisfying this
condition, and they are denoted here v'(k), for k=0,1...N_connected.

|Argument|            Type            |                   Description                    |
|--------|----------------------------|--------------------------------------------------|
|v       |numpy.ndarray[float64[m, 1]]|A constant reference to the visible configuration.|

### to_dense
Returns the dense matrix representation of the operator. Note that, in general,
the size of the matrix is exponential in the number of quantum
numbers, and this operation should thus only be performed for
low-dimensional Hilbert spaces.

This method requires an indexable Hilbert space.



### to_sparse
Returns the sparse matrix representation of the operator. Note that, in general,
the size of the matrix is exponential in the number of quantum
numbers, and this operation should thus only be performed for
low-dimensional Hilbert spaces or sufficiently sparse operators.

This method requires an indexable Hilbert space.



### transpose
Returns the transpose of this operator


## Properties

|   Property   |         Type         |                    Description                     |
|--------------|----------------------|----------------------------------------------------|
|acting_on     |list[list]            | A list of the sites that each local matrix acts on.|
|hilbert       |netket.hilbert.Hilbert| ``Hilbert`` space of operator.                     |
|local_matrices|list[list]            | A list of the local matrices.                      |
