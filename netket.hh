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

#ifndef NETKET_HEADER_HH
#define NETKET_HEADER_HH

#include <random>

namespace netket {
using default_random_engine = std::mt19937;
}

#include "Graph/graph.hh"
#include "Hamiltonian/hamiltonian.hh"
#include "Hilbert/hilbert.hh"
#include "Learning/learning.hh"
#include "Lookup/lookup.hh"
#include "Machine/machine.hh"
#include "Observable/observable.hh"
#include "Parallel/parallel.hh"
#include "Sampler/sampler.hh"
#include "Stats/stats.hh"
#include "Json/json.hh"

#endif
