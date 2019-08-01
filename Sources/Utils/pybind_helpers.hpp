#ifndef NETKET_PYBIND_HELPERS_HPP
#define NETKET_PYBIND_HELPERS_HPP

#include <pybind11/eigen.h>
#include <pybind11/eval.h>
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
Value GetOrDefault(const pybind11::kwargs &kwargs, std::string field,
                   Value defval) {
  if (kwargs.contains(field.c_str())) {
    return pybind11::cast<Value>(kwargs[field.c_str()]);
  } else {
    return defval;
  }
}

template <class Value>
Value GetOrThrow(const pybind11::kwargs &kwargs, std::string field) {
  if (kwargs.contains(field.c_str())) {
    return pybind11::cast<Value>(kwargs[field.c_str()]);
  } else {
    std::stringstream str;
    str << "Key not found in keyword arguments: " << field;
    throw InvalidInputError(str.str());
  }
}

}  // namespace netket

/// \brief C++11 implementation of C++14 std::integer_sequence.
///
/// See https://en.cppreference.com/w/cpp/utility/integer_sequence.
template <class T, T... Is>
struct integer_sequence {
  typedef T value_type;
  static_assert(
      std::is_integral<T>::value,
      "integer_sequence can only be instantiated with an integral type");
  static constexpr size_t size() noexcept { return sizeof...(Is); }
};

/// \brief C++11 implementation of C++14 std::index_sequence
template <size_t... Is>
using index_sequence = integer_sequence<size_t, Is...>;

namespace detail {
template <size_t N, size_t... Is>
struct make_index_sequence_impl
    : public make_index_sequence_impl<N - 1, N - 1, Is...> {};

template <size_t... Is>
struct make_index_sequence_impl<0, Is...>
    : public integer_sequence<size_t, Is...> {};
}  // namespace detail

/// \brief C++11 implementation of C++14 std::make_index_sequence
template <size_t N>
using make_index_sequence = detail::make_index_sequence_impl<N>;

namespace detail {
/// \brief A helper functor for implementing StateDict functions.
struct ToStateDictFn {
 private:
  template <class EigenObject>
  pybind11::object CastOne(
      const std::pair<const char *, EigenObject> &item) const {
    // NOTE: pybind11's C-style string type_caster uses std::string type_caster
    // which in turn ignores return_value_policy (i.e. it always copies data),
    // so it's safe to use reference here since it will only be applied to Eigen
    // types.
    return pybind11::cast(item, pybind11::return_value_policy::reference);
  }

  template <class State, size_t... Is>
  pybind11::list operator()(State &&state, index_sequence<Is...>) const {
    pybind11::list obj;
    __attribute__((unused)) auto _dummy = {
        (obj.append(CastOne(std::get<Is>(std::forward<State>(state)))), 0)...};
    return obj;
  }

 public:
  template <class State>
  pybind11::list operator()(State &&state) const {
    return (*this)(std::forward<State>(state),
                   make_index_sequence<std::tuple_size<
                       typename std::remove_reference<State>::type>::value>{});
  }
};
}  // namespace detail

/// \brief A helper functor for implementing AbstractMachine::StateDict
/// function.
///
/// You pass it a tuple, e.g. for an RBM:
///
///     std::make_tuple(
///         std::make_pair("a", a_),
///         std::make_pair("b", b_),
///         std::make_pair("W", W_)
///     )
///
/// and it automatically converts it to Python OrderedDict from collections
/// module. Order of elements in the dictionary will be the same as the order of
/// elements in the tuple.
template <class Tuple>
PyObject *ToStateDict(Tuple &&state) {
  auto OrderedDict =
      pybind11::module::import("collections").attr("OrderedDict");
  return OrderedDict(detail::ToStateDictFn{}(std::forward<Tuple>(state)))
      .release()
      .ptr();
}

#endif  // NETKET_PYBIND_HELPERS_HPP
