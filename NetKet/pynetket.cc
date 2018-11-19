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

#ifndef NETKET_PYNETKET_CC
#define NETKET_PYNETKET_CC

#include <netket.hpp>
namespace netket {
using MachineType = std::complex<double>;
using AbMachineType = AbstractMachine<MachineType>;
using AbLayerType = AbstractLayer<MachineType>;
using AbSamplerType = AbstractSampler<AbMachineType>;
}  // namespace netket

#include "Dynamics/pydynamics.hpp"
#include "Graph/pygraph.hpp"
#include "GroundState/pyground_state.hpp"
#include "Hilbert/pyhilbert.hpp"
#include "Machine/pymachine.hpp"
#include "Operator/pyoperator.hpp"
#include "Optimizer/pyoptimizer.hpp"
#include "Output/pyoutput.hpp"
#include "Sampler/pysampler.hpp"
#include "Stats/binning.hpp"
#include "Utils/pyutils.hpp"

namespace netket {

namespace detail {
// TODO(twesterhout): Strictly speaking, this is unsafe, I'm afraid, because one
// can load the shared library from two processes which could result in MPI not
// being initialized or not finalized twice, but come on... let's hope (for now)
// noone is their sane mind is goind to do that.
struct MPIInitializer {
  MPIInitializer() {
    int already_initialized;
    MPI_Initialized(&already_initialized);
    if (!already_initialized) {
      // We don't have access to command-line arguments
      if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) {
        std::ostringstream msg;
        msg << "This should never have happened. How did you manage to "
               "call MPI_Init() in between two C function calls?! "
               "Terminating now.";
        std::cerr << msg.str() << std::endl;
        std::terminate();
      }
      have_initialized_ = true;
#if !defined(NDEBUG)
      std::cerr << "MPI successfully initialized by NetKet." << std::endl;
#endif
    }
  }

  ~MPIInitializer() {
    if (have_initialized_) {
      // We have initialized MPI so it's only right we finalize it.
      MPI_Finalize();
#if !defined(NDEBUG)
      std::cerr << "MPI successfully finalized by NetKet." << std::endl;
#endif
    }
  }

 private:
  bool have_initialized_;
};
static MPIInitializer _do_not_use_me_dummy_{};
} // namespace detail

using ode::AddDynamicsModule;

PYBIND11_MODULE(netket, m) {
  AddDynamicsModule(m);
  AddGraphModule(m);
  AddGroundStateModule(m);
  AddHilbertModule(m);
  AddMachineModule(m);
  AddOperatorModule(m);
  AddOptimizerModule(m);
  AddOutputModule(m);
  AddSamplerModule(m);
  AddUtilsModule(m);
}  // PYBIND11_MODULE

}  // namespace netket

#endif
