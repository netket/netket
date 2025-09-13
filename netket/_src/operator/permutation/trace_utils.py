# Copyright 2025 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Utility functions for computing traces of permutation operators

import numpy as np

from itertools import product


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


def get_subset_occupations(partition_labels, subsets):
    """
    Given a list of subsets of [0, ..., n-1] and a partition of [0, ..., n-1],
    return a list of the number of element in each partition for each subset.

    Args:
        partition_labels (<int>): The list such that partition_labels[k] is the partition to which k belongs.
        subsets (<<int>>): The list of subsets.

    Return:
        <ndarray>: The list such that l[i][j] is the number of elements of subsets[i] in partition j.
    """
    n_partitions = len(np.unique(partition_labels))
    occupations = []
    for subset in subsets:
        occupation_count = np.zeros(n_partitions, dtype=int)
        for k in subset:
            occupation_count[partition_labels[k]] += 1
        occupations.append(occupation_count)
    return occupations


def get_parity_sum(occupation_list, n_occupations):
    """
    Given a list occupation_list of lists of length p and n_occupations, a list of length p,
    we look at all subsets of occupation_list such that the sum of its elements is n_occupations.
    We return the sum over all such subsets, of the parity of the number of indices k in that subset such
    that sum(occupation_list[k]) is even.

    Args:
        occupation_list (<ndarray>): The list of lists of length p.
        n_occupations (<int>): The list of target occupation.

    Return:
        int: The sum of parities specified above.
    """

    table = np.full(
        (
            len(occupation_list) + 1,
            *(n_occupation + 1 for n_occupation in n_occupations),
        ),
        -1,
    )

    table[0] = 0
    table[0][(0,) * len(n_occupations)] = 1

    for i in range(len(occupation_list)):

        occupation_iterator = product(
            *list(range(n_occupation + 1) for n_occupation in n_occupations)
        )

        for occupation in occupation_iterator:

            not_included_sum = np.array(occupation) - occupation_list[i]
            if np.any(not_included_sum < 0):
                table[i + 1][occupation] = table[i][occupation]
            else:
                table[i + 1][occupation] = (-1) ** (
                    sum(occupation_list[i]) + 1
                ) * table[i][tuple(not_included_sum)] + table[i][occupation]

    return table[-1][tuple(n_occupations)].item()
