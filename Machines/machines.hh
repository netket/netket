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

#ifndef NETKET_MACHINES_HH
#define NETKET_MACHINES_HH

/** @defgroup machines Machines module
 *
 * Machines module contains implementations of wave-functions. 
 */

namespace netket{
  template<class T> class AbstractMachine;
  template<class T> class RbmSpin;
  template<class T> class RbmSpinSymm;
  template<class T> class RbmMultival;

  template<class T> class Machine;
}

#include "abstract_machine.hh"

#include "rbm_spin.hh"
#include "rbm_spin_symm.hh"
#include "rbm_multival.hh"

#include "machine.cc"
#endif
