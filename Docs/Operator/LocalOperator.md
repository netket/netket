# LocalOperator
A custom local operator.
## Constructor [1]
Constructs a new ``LocalOperator`` given a hilbert space and (if
specified) a constant level shift.

| Field  |         Type         |               Description               |
|--------|----------------------|-----------------------------------------|
|hilbert |netket.hilbert.Hilbert|Hilbert space the operator acts on.      |
|constant|float=0.0             |Level shift for operator. Default is 0.0.|
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

## Constructor [2]
Constructs a new ``LocalOperator`` given a hilbert space, a vector of
operators, a vector of sites, and (if specified) a constant level
shift.

|  Field  |          Type           |                       Description                        |
|---------|-------------------------|----------------------------------------------------------|
|hilbert  |netket.hilbert.Hilbert   |Hilbert space the operator acts on.                       |
|operators|List[List[List[complex]]]|A list of operators, in matrix form.                      |
|acting_on|List[List[int]]          |A list of sites, which the corresponding operators act on.|
|constant |float=0.0                |Level shift for operator. Default is 0.0.                 |
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

## Constructor [3]
Constructs a new ``LocalOperator`` given a hilbert space, an
operator, a site, and (if specified) a constant level
shift.

|  Field  |         Type         |                       Description                        |
|---------|----------------------|----------------------------------------------------------|
|hilbert  |netket.hilbert.Hilbert|Hilbert space the operator acts on.                       |
|operator |List[List[complex]]   |An operator, in matrix form.                              |
|acting_on|List[int]             |A list of sites, which the corresponding operators act on.|
|constant |float=0.0             |Level shift for operator. Default is 0.0.                 |
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


## Properties
|   Property   |         Type         |                    Description                     |
|--------------|----------------------|----------------------------------------------------|
|acting_on     |list[list]            | A list of the sites that each local matrix acts on.|
|hilbert       |netket.hilbert.Hilbert| ``Hilbert`` space of operator.                     |
|local_matrices|list[list]            | A list of the local matrices.                      |

