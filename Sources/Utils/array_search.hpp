// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_ARRAYSEARCH_HPP
#define NETKET_ARRAYSEARCH_HPP

#include "Utils/exceptions.hpp"

namespace netket {

/**
  Returns the smallest non zero element of an array.
*/
template <class ForwardIterator>
ForwardIterator min_nonzero_elem(ForwardIterator first, ForwardIterator last) {
  if (first == last)
    throw RuntimeError{"Error: first and last iterator are the same.\n"};

  ForwardIterator smallest = first;
  while (++first != last)
    if (*first < *smallest && *first != 0) smallest = first;
  assert(smallest != last);
  assert(*smallest != 0 && "There aren't non-zero elements.\n");
  return smallest;
}

}  // namespace netket

#endif
