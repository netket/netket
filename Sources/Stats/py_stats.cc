#include "Stats/stats.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Stats/mc_stats.hpp"
#include "Stats/obs_manager.hpp"
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

  subm.def("statistics", &Statistics, py::arg{"values"}, py::arg{"n_chains"},
           R"EOF(Computes some statistics (see `Stats` class) of a sequence of
                 local estimators obtained from Monte Carlo sampling.

                 Args:
                     values: A vector of local estimators.
                     n_chains: Number of chains interleaved in `values`.)EOF");
}
}  // namespace detail
}  // namespace netket

namespace netket {

void AddStatsModule(PyObject* m) {
  detail::AddStatsModule(py::module{py::reinterpret_borrow<py::object>(m)});
}

}  // namespace netket
