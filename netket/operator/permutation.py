__all__ = [
    "PermutationOperator",
    "PermutationOperatorFermion",
    "construct_permutation_operator",
]

from netket._src.operator.permutation.spin import (
    PermutationOperator as PermutationOperator,
)
from netket._src.operator.permutation.fermion import (
    PermutationOperatorFermion as PermutationOperatorFermion,
)
from netket._src.operator.permutation.construct import (
    construct_permutation_operator as construct_permutation_operator,
)
