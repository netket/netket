import numpy as np


def count_n_uplets(values, target, m):
    r"""
    Count the number of n-uplets $s \in \{0, ..., m-1\}^n$ such that
    $$ \sum_{k=0}^{n-1} 2 s_k a_k = t $$
    where $a_k$ corresponds to `values[k]` and $t$ to `target`.

    Args:
        values (<int>): The list of n integers.
        target (int): The target value for the sums.
        m (int): The number of choice per index of values.

    Return:
        int: The number of such n-uplets.
    """

    table = np.full((len(values) + 1, target + 1), -1)
    table[0] = 0
    table[0][0] = 1

    for i in range(len(values)):
        for j in range(table.shape[1]):
            accumulated_value = 0
            # Add the contribution from element i being added at magnitude k
            for k in range(m):
                not_included_sum = j - 2 * k * values[i]
                if not_included_sum >= 0:
                    accumulated_value += table[i][not_included_sum]
                table[i + 1][j] = accumulated_value

    return table[-1][-1]
