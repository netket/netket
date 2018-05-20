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

#ifndef NETKET_NEXT_VARIATION_HPP
#define NETKET_NEXT_VARIATION_HPP

#include <iterator>

namespace netket{

  //From StackOverflow, Matteo Gattanini's code
  // Variations with repetition in lexicographic order
  // k: length of alphabet (available symbols)
  // n: number of places
  // The number of possible variations (cardinality) is k^n (it's like counting)
  // Sequence elements must be comparable and increaseable (operator<, operator++)
  // The elements are associated to values 0÷(k-1), max=k-1
  // The iterators are at least bidirectional and point to the type of 'max'
  template <class Iter>
  bool next_variation(Iter first, Iter last, const typename std::iterator_traits<Iter>::value_type max)
  {
    if(first == last) return false; // empty sequence (n==0)

    Iter i(last); --i; // Point to the rightmost element
    // Check if I can just increase it
    if(*i < max) { ++(*i); return true; } // Increase this element and return

    // Find the rightmost element to increase
    while( i != first )
    {
      *i = 0; // reset the right-hand element
      --i; // point to the left adjacent
      if(*i < max) { ++(*i); return true; } // Increase this element and return
    }

    return false;
  }

}

#endif
