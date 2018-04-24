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

#ifndef NETKET_HAMILTONIAN_HH
#define NETKET_HAMILTONIAN_HH

namespace netket{
  class AbstractHamiltonian;
  template<class G> class Ising;
  template<class G> class Heisenberg;
  template<class G> class BoseHubbard;
  class CustomHamiltonian;
  template<class G> class Hamiltonian;
}

#include "abstract_hamiltonian.hh"
#include "ising.hh"
#include "heisenberg.hh"
#include "bosonhubbard.hh"
#include "custom_hamiltonian.hh"
#include "hamiltonian.cc"
#endif
