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

#ifndef NETKET_SAMPLER_HH
#define NETKET_SAMPLER_HH

namespace netket{
  template<class WfType> class AbstractSampler;
  template<class WfType> class Metropolis;
  template<class WfType> class MetropolisLocal;
  template<class WfType> class MetropolisLocalPt;
  template<class WfType> class MetropolisExchange;
  template<class WfType> class MetropolisExchangePt;
  template<class WfType> class MetropolisHop;
  template<class WfType,class HamType> class MetropolisHamiltonian;
  template<class WfType,class HamType> class MetropolisHamiltonianPt;
  template<class WfType> class Sampler;
}

#include "abstract_sampler.hh"
#include "metropolis_local.hh"
#include "metropolis_exchange.hh"
#include "metropolis_exchange_pt.hh"
#include "metropolis_local_pt.hh"
#include "metropolis_hop.hh"
#include "metropolis_hamiltonian.hh"
#include "metropolis_hamiltonian_pt.hh"
#include "sampler.cc"
#endif
