#include "Stats/stats.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Stats/mc_stats.hpp"
#include "Stats/obs_manager.hpp"
#include "Utils/exceptions.hpp"
#include "common_types.hpp"

namespace py = pybind11;

namespace netket {
namespace detail {

py::dict GetItem(const ObsManager& self, const std::string& name) {
  py::dict dict;
  self.InsertAllStats(name, dict);
  return dict;
}

void AddStatsModule(py::module m) {
  auto subm = m.def_submodule("stats");

  py::class_<ObsManager>(subm, "ObsManager")
      .def("__getitem__", &detail::GetItem, py::arg("name"))
      .def("__getattr__", &detail::GetItem, py::arg("name"))
      .def("__contains__", &ObsManager::Contains, py::arg("name"))
      .def("__len__", &ObsManager::Size)
      .def("keys", &ObsManager::Names)
      .def("__repr__", [](const ObsManager& self) {
        std::string s("<netket.stats.ObsManager: size=");
        auto size = self.Size();
        s += std::to_string(size);
        if (size > 0) {
          s += " [";
          for (const auto& name : self.Names()) {
            s += name + ", ";
          }
          // remove last comma + space:
          s.pop_back();
          s.pop_back();

          s += "]";
        }
        return s + ">";
      });

  subm.def(
      "statistics",
      [](py::array_t<Complex, py::array::c_style> local_values) {
        switch (local_values.ndim()) {
          case 2:
            return Statistics(
                Eigen::Map<const Eigen::VectorXcd>{
                    local_values.data(),
                    local_values.shape(0) * local_values.shape(1)},
                local_values.shape(1));
          case 1:
            return Statistics(
                Eigen::Map<const Eigen::VectorXcd>{local_values.data(),
                                                   local_values.shape(0)},
                1);
          default:
            NETKET_CHECK(false, InvalidInputError,
                         "local_values has wrong dimension: "
                             << local_values.ndim()
                             << "; expected either 1 or 2.");
        }  // end switch
      },
      py::arg{"values"}.noconvert(),
      R"EOF(Computes some statistics (see `Stats` class) of a sequence of
                 local estimators obtained from Monte Carlo sampling.

                 Args:
                     values: A tensor of local estimators.)EOF");
}
}  // namespace detail
}  // namespace netket

namespace netket {

void AddStatsModule(PyObject* m) {
  detail::AddStatsModule(py::module{py::reinterpret_borrow<py::object>(m)});
}

}  // namespace netket
