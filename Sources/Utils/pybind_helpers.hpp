#ifndef NETKET_PYBIND_HELPERS_HPP
#define NETKET_PYBIND_HELPERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nonstd/optional.hpp>

#include "exceptions.hpp"

// This adds pybind11 support for nonstd::optional, see
// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<nonstd::optional<T>>
    : public optional_caster<nonstd::optional<T>> {};

template <>
struct type_caster<nonstd::nullopt_t> : public void_caster<nonstd::nullopt_t> {
};

}  // namespace detail
}  // namespace pybind11

namespace netket {

template <class Value>
Value GetOrDefault(const pybind11::kwargs& kwargs, std::string field,
                   Value defval) {
  if (kwargs.contains(field.c_str())) {
    return pybind11::cast<Value>(kwargs[field.c_str()]);
  } else {
    return defval;
  }
}

template <class Value>
Value GetOrThrow(const pybind11::kwargs& kwargs, std::string field) {
  if (kwargs.contains(field.c_str())) {
    return pybind11::cast<Value>(kwargs[field.c_str()]);
  } else {
    std::stringstream str;
    str << "Key not found in keyword arguments: " << field;
    throw InvalidInputError(str.str());
  }
}

}  // namespace netket

#endif  // NETKET_PYBIND_HELPERS_HPP
