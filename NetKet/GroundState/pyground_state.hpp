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

#ifndef NETKET_PYGROUND_STATE_HPP
#define NETKET_PYGROUND_STATE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ground_state.hpp"

namespace py = pybind11;

namespace netket {

namespace detail {
/**
 * Wraps an iterator over simulation steps (like
 * VariationalMonteCarlo::Iterator) and dereferences to a Python dictionary.
 * @tparam It The wrapped C++ iterator type.
 */
template <class It>
class PyIteratorAdaptor {
 public:
  // typedefs required for iterators
  using iterator_category = std::input_iterator_tag;
  using difference_type = Index;
  using value_type = py::dict;
  using pointer_type = py::dict *;
  using reference_type = py::dict &;

  explicit PyIteratorAdaptor<It>(It it) : it_(std::move(it)) {}

  bool operator!=(const PyIteratorAdaptor<It> &other) {
    return it_ != other.it_;
  }
  bool operator==(const PyIteratorAdaptor<It> &other) {
    return it_ == other.it_;
  }

  PyIteratorAdaptor<It> operator++() {
    ++it_;
    return *this;
  }

  py::dict operator*() {
    auto step = *it_;
    py::dict dict;
    step.observables.InsertAllStats(dict);
    return dict;
  }

  PyIteratorAdaptor<It> begin() const { return *this; }
  PyIteratorAdaptor<It> end() const { return *this; }

 private:
  It it_;
};
}  // namespace detail

void AddGroundStateModule(py::module &m) {
  auto m_exact = m.def_submodule("exact");
  auto m_vmc = m.def_submodule("vmc");

  py::class_<VariationalMonteCarlo>(m_vmc, "Vmc")
      .def(py::init<const AbstractOperator &, SamplerType &,
                    AbstractOptimizer &, int, int, int, const std::string &,
                    double, bool, bool, bool>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>(), py::arg("hamiltonian"), py::arg("sampler"),
           py::arg("optimizer"), py::arg("n_samples"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0, py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("rescale_shift") = false,
           py::arg("use_iterative") = false, py::arg("use_cholesky") = true)
      .def_property_readonly("machine", &VariationalMonteCarlo::GetMachine)
      .def("add_observable", &VariationalMonteCarlo::AddObservable,
           py::keep_alive<1, 2>())
      .def("run", &VariationalMonteCarlo::Run, py::arg("output_prefix"),
           py::arg("max_steps") = nonstd::nullopt, py::arg("step_size") = 1,
           py::arg("save_params_every") = 50)
      .def("iter", &VariationalMonteCarlo::Iterate,
           py::arg("max_steps") = nonstd::nullopt, py::arg("step_size") = 1);

  py::class_<VariationalMonteCarlo::Iterator>(m_vmc, "Iterator")
      .def("__iter__", [](VariationalMonteCarlo::Iterator &self) {
        detail::PyIteratorAdaptor<VariationalMonteCarlo::Iterator> it(self);
        return py::make_iterator(it.begin(), it.end());
      });

  py::class_<ImagTimePropagation>(m_exact, "ImagTimePropagation")
      .def(py::init<ImagTimePropagation::Matrix &,
                    ImagTimePropagation::Stepper &, double,
                    ImagTimePropagation::StateVector>(),
           py::arg("hamiltonian"), py::arg("stepper"), py::arg("t0"),
           py::arg("initial_state"))
      .def("add_observable", &ImagTimePropagation::AddObservable,
           py::keep_alive<1, 2>(), py::arg("observable"), py::arg("name"),
           py::arg("matrix_type") = "Sparse")
      .def("iter", &ImagTimePropagation::Iterate, py::arg("dt"),
           py::arg("max_steps") = nonstd::nullopt,
           py::arg("store_state") = true)
      .def_property("t", &ImagTimePropagation::GetTime,
                    &ImagTimePropagation::SetTime);

  py::class_<ImagTimePropagation::Iterator>(m_exact, "ImagTimeIterator")
      .def("__iter__", [](ImagTimePropagation::Iterator &self) {
        detail::PyIteratorAdaptor<ImagTimePropagation::Iterator> it(self);
        return py::make_iterator(it.begin(), it.end());
      });

  py::class_<eddetail::result_t>(m_exact, "EdResult")
      .def_readwrite("eigenvalues", &eddetail::result_t::eigenvalues)
      .def_readwrite("eigenvectors", &eddetail::result_t::eigenvectors)
      .def_readwrite("which_eigenvector",
                     &eddetail::result_t::which_eigenvector);

  m_exact.def("LanczosEd", &lanczos_ed, py::arg("operator"),
              py::arg("matrix_free") = false, py::arg("first_n") = 1,
              py::arg("max_iter") = 1000, py::arg("seed") = 42,
              py::arg("precision") = 1.0e-14,
              py::arg("get_groundstate") = false);
}

}  // namespace netket

#endif
