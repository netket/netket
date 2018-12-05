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

#ifndef NETKET_PYOUTPUT_HPP
#define NETKET_PYOUTPUT_HPP

#include <pybind11/pybind11.h>

#include "json_output_writer.hpp"

namespace py = pybind11;

namespace netket {

void AddOutputModule(py::module &m) {
  auto subm = m.def_submodule("output");

  py::class_<JsonOutputWriter>(subm, "JsonOutputWriter")
      .def(py::init<const std::string &, const std::string &, int>(),
           py::arg("log_file_name"), py::arg("wavefunc_file_name"),
           py::arg("save_every") = 50);
}

}  // namespace netket

#endif
