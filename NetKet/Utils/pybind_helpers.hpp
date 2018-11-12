#ifndef NETKET_PYBIND_HELPERS_HPP
#define NETKET_PYBIND_HELPERS_HPP

#include <pybind11/pybind11.h>

#include "exceptions.hpp"

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
