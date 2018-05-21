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

#ifndef NETKET_HEADER_HPP
#define NETKET_HEADER_HPP

#include <random>

namespace netket {
using default_random_engine = std::mt19937;
}

#include "Graph/graph.hpp"
#include "Hamiltonian/hamiltonian.hpp"
#include "Headers/welcome.hpp"
#include "Hilbert/hilbert.hpp"
#include "Learning/learning.hpp"
#include "Lookup/lookup.hpp"
#include "Machine/machine.hpp"
#include "Observable/observable.hpp"
#include "Parallel/parallel.hpp"
#include "Sampler/sampler.hpp"
#include "Stats/stats.hpp"
#include "Json/json_helper.hpp"

#endif
