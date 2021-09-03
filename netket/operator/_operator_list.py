from typing import Union
import numpy as np


class OperatorList:
    """
    Data class to be used internally to handle the matrix representation of
    different operators. This class helps in storing only the necessary
    matrices in memory. The instances of this class behave like a list,
    but they only store unrepeated operators.
    """

    def __init__(self):
        self._operators = {}
        self._index2operatorkeys = []

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        if isinstance(key, slice):
            length = len(self._index2operatorkeys)
            return [
                self._operators[self._index2operatorkeys[k]]
                for k in range(*key.indices(length))
            ]
        return self._operators[self._index2operatorkeys[key]]

    def __setitem__(self, key: int, operator: np.ndarray) -> None:
        value_hash = hash(operator.tostring())
        self._index2operatorkeys[key] = value_hash
        if value_hash not in self._operators.keys():
            self._operators[value_hash] = operator

    def __delitem__(self, key: Union[int, slice]) -> None:
        del self._index2operatorkeys[key]

    def __iter__(self):
        self._n_iter = 0
        return self

    def __next__(self):
        length = len(self._index2operatorkeys)
        if self._n_iter < length:
            result = self._operators[self._index2operatorkeys[self._n_iter]]
            self._n_iter += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        return f"OperatorList(n_operators={len(self._index2operatorkeys)}, n_unique_operators={len(self._operators)})"

    def append(self, operator: np.ndarray) -> None:
        initial_length = len(self._index2operatorkeys)
        self._index2operatorkeys.append(0)  # dummy value
        self[initial_length] = operator
