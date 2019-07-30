#ifndef NETKET_PYSTATS_HPP
#define NETKET_PYSTATS_HPP

#include <Python.h>
#include <pybind11/pybind11.h>

// TODO: Move me to .cc file!
// Expose the Stats object to Python as dict
namespace pybind11 {
namespace detail {

using NkStatsType = netket::Binning<double>::Stats;

template <>
struct type_caster<NkStatsType> {
 public:
  PYBIND11_TYPE_CASTER(NkStatsType, _("_Stats"));

  static handle cast(NkStatsType src, return_value_policy /* policy */,
                     handle /* parent */) {
    py::dict dict;
    dict["Mean"] = src.mean;
    dict["Sigma"] = src.sigma;
    dict["Taucorr"] = src.taucorr;
    return dict.release();
  }
};

}  // namespace detail
}  // namespace pybind11

namespace netket {

void AddStatsModule(PyObject* module);

}  // namespace netket

#endif  // NETKET_PYSTATS_HPP
