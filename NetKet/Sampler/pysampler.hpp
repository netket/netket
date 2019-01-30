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

#ifndef NETKET_PYSAMPLER_HPP
#define NETKET_PYSAMPLER_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "Graph/graph.hpp"
#include "Operator/operator.hpp"
#include "Utils/memory_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "abstract_sampler.hpp"
#include "custom_sampler.hpp"
#include "custom_sampler_pt.hpp"
#include "exact_sampler.hpp"
#include "metropolis_exchange.hpp"
#include "metropolis_exchange_pt.hpp"
#include "metropolis_hamiltonian.hpp"
#include "metropolis_hamiltonian_pt.hpp"
#include "metropolis_hop.hpp"
#include "metropolis_local.hpp"
#include "metropolis_local_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddSamplerModule(py::module &m) {
  auto subm = m.def_submodule("sampler");

  py::class_<SamplerType>(subm, "Sampler", R"EOF(
    NetKet implements generic sampling routines to be used in conjunction with
    suitable variational states, the `Machines`.
    A `Sampler` generates quantum numbers distributed according to the square modulus
    of the wave-function:

    $$P(s_1\dots s_N) = |\Psi(s_1\dots s_N) | ^2.$$

    The samplers typically transit from the current set of quantum numbers
    $$\mathbf{s} = s_1 \dots s_N$$ to another set
    $$\mathbf{s^\prime} = s^\prime_1 \dots s^\prime_N$$.
    Samplers are then fully specified by the transition probability:

    $$T( \mathbf{s} \rightarrow \mathbf{s}^\prime) .$$
    )EOF")
      .def("seed", &SamplerType::Seed, py::arg("base_seed"), R"EOF(
      Seeds the random number generator used by the ``Sampler``.

      Args:
          base_seed: The base seed for the random number generator
          used by the sampler. Each MPI node is guarantueed to be initialized
          with a distinct seed.
      )EOF")
      .def("reset", &SamplerType::Reset, py::arg("init_random") = false, R"EOF(
      Resets the state of the sampler, including the acceptance rate statistics
      and optionally initializing at random the visible units being sampled.

      Args:
          init_random: If ``True`` the quantum numbers (visible units)
          are initialized at random, otherwise their value is preserved.
      )EOF")
      .def("sweep", &SamplerType::Sweep, R"EOF(
      Performs a sampling sweep. Typically a single sweep
      consists of an extensive number of local moves.
      )EOF")
      .def_property("visible", &SamplerType::Visible, &SamplerType::SetVisible,
                    R"EOF(
                      numpy.array: The quantum numbers being sampled,
                       and distributed according to $$|\Psi(v)|^2$$ )EOF")
      .def_property_readonly("acceptance", &SamplerType::Acceptance, R"EOF(
        numpy.array: The measured acceptance rate for the sampling.
        In the case of rejection-free sampling this is always equal to 1.  )EOF")
      .def_property_readonly("hilbert", &SamplerType::GetHilbert, R"EOF(
        netket.hilbert: The Hilbert space used for the sampling.  )EOF")
      .def_property_readonly("machine", &SamplerType::GetMachine, R"EOF(
        netket.machine: The machine used for the sampling.  )EOF");

  {
    using DerSampler = MetropolisLocal<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisLocal", R"EOF(
      This sampler acts locally only on one local degree of freedom $$s_i$$,
      and proposes a new state: $$ s_1 \dots s^\prime_i \dots s_N $$,
      where $$ s^\prime_i \neq s_i $$.

      The transition probability associated to this
      sampler can be decomposed into two steps:

      1. One of the site indices $$ i = 1\dots N $$ is chosen
      with uniform probability.
      2. Among all the possible ($$m$$) values that $$s_i$$ can take,
      one of them is chosen with uniform probability.

      For example, in the case of spin $$1/2$$ particles, $$m=2$$
      and the possible local values are $$s_i = -1,+1$$.
      In this case then `MetropolisLocal` is equivalent to flipping a random spin.

      In the case of bosons, with occupation numbers
      $$s_i = 0, 1, \dots n_{\mathrm{max}}$$, `MetropolisLocal`
      would pick a random local occupation number uniformly between $$0$$
      and $$n_{\mathrm{max}}$$.
      )EOF")
        .def(py::init<MachineType &>(), py::keep_alive<1, 2>(),
             py::arg("machine"), R"EOF(
             Constructs a new ``MetropolisLocal`` sampler given a machine.

             Args:
                 machine: A machine used for the sampling.
                      The probability distribution being sampled
                      from is $$|\Psi(s)|^2$$.

             Examples:
                 Sampling from a RBM machine in a 1D lattice of spin 1/2

                 ```python
                 >>> import netket as nk
                 >>> from mpi4py import MPI
                 >>>
                 >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
                 >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
                 >>>
                 >>> # RBM Spin Machine
                 >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
                 >>>
                 >>> # Construct a MetropolisLocal Sampler
                 >>> sa = nk.sampler.MetropolisLocal(machine=ma)
                 >>> print(sa.hilbert.size)
                 100

                 ```
             )EOF");
  }

  {
    using DerSampler = MetropolisLocalPt<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisLocalPt")
        .def(py::init<MachineType &, int>(), py::keep_alive<1, 2>(),
             py::arg("machine"), py::arg("n_replicas") = 1);
  }

  {
    using DerSampler = MetropolisHop<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisHop")
        .def(py::init<MachineType &, int>(), py::keep_alive<1, 3>(),
             py::arg("machine"), py::arg("d_max") = 1);
  }

  {
    using DerSampler = MetropolisHamiltonian<MachineType, AbstractOperator>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisHamiltonian")
        .def(py::init<MachineType &, AbstractOperator &>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
             py::arg("hamiltonian"));
  }

  {
    using DerSampler = MetropolisHamiltonianPt<MachineType, AbstractOperator>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisHamiltonianPt")
        .def(py::init<MachineType &, AbstractOperator &, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
             py::arg("hamiltonian"), py::arg("n_replicas"));
  }

  {
    using DerSampler = MetropolisExchange<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisExchange")
        .def(py::init<const AbstractGraph &, MachineType &, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max") = 1);
  }

  {
    using DerSampler = MetropolisExchangePt<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "MetropolisExchangePt")
        .def(py::init<const AbstractGraph &, MachineType &, int, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max") = 1,
             py::arg("n_replicas") = 1);
  }

  {
    using DerSampler = ExactSampler<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "ExactSampler")
        .def(py::init<MachineType &>(), py::keep_alive<1, 2>(),
             py::arg("machine"));
  }

  {
    using DerSampler = CustomSampler<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "CustomSampler")
        .def(py::init<MachineType &, const LocalOperator &,
                      const std::vector<double> &>(),
             py::keep_alive<1, 2>(), py::arg("machine"),
             py::arg("move_operators"),
             py::arg("move_weights") = std::vector<double>());
  }

  {
    using DerSampler = CustomSamplerPt<MachineType>;
    py::class_<DerSampler, SamplerType>(subm, "CustomSamplerPt")
        .def(py::init<MachineType &, const LocalOperator &,
                      const std::vector<double> &, int>(),
             py::keep_alive<1, 2>(), py::arg("machine"),
             py::arg("move_operators"),
             py::arg("move_weights") = std::vector<double>(),
             py::arg("n_replicas") = 1);
  }
}

}  // namespace netket

#endif
