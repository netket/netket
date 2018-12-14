#ifndef NETKET_PYBIND_SUPPORT_HPP
#define NETKET_PYBIND_SUPPORT_HPP

#include <nonstd/optional.hpp>
#include <pybind11/stl.h>

// This adds pybind11 support for nonstd::optional, see
// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<nonstd::optional<T>> : optional_caster<nonstd::optional<T>> {
};
}  // namespace detail
}  // namespace pybind11

#endif //NETKET_PYBIND_SUPPORT_HPP
