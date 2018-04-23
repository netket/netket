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

#ifndef NETKET_HILBERT_HH
#define NETKET_HILBERT_HH

namespace netket{
  class AbstractHilbert;
  class Spin;
  class Boson;
  class Qubit;
  class CustomHilbert;
  class Hilbert;
  class LocalOperator;
}

#include "abstract_hilbert.hh"
#include "next_variation.hh"
#include "spins.hh"
#include "bosons.hh"
#include "qubits.hh"
#include "custom_hilbert.hh"
#include "hilbert.cc"
#include "local_operator.hh"

#endif
