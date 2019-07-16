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

#ifndef NETKET_EXCEPTIONS_HPP
#define NETKET_EXCEPTIONS_HPP

#include <stdexcept>
#include <tuple>

namespace netket {

using InvalidInputError = std::invalid_argument;
using RuntimeError = std::runtime_error;

namespace detail {
[[noreturn]] void WrongShape(const char *func, const char *arg, long shape,
                             long expected);
[[noreturn]] void WrongShape(const char *func, const char *arg,
                             std::pair<long, long> const &shape,
                             std::pair<long, decltype(std::ignore)> expected);
[[noreturn]] void WrongShape(const char *func, const char *arg,
                             std::pair<long, long> const &shape,
                             std::pair<decltype(std::ignore), long> expected);
[[noreturn]] void WrongShape(const char *func, const char *arg,
                             std::pair<long, long> const &shape,
                             std::pair<long, long> expected);
}  // namespace detail

inline void CheckShape(const char *func, const char *arg, long shape,
                       long expected) {
  if (shape != expected) {
    detail::WrongShape(func, arg, shape, expected);
  }
}

inline void CheckShape(const char *func, const char *arg,
                       std::pair<long, long> const &shape,
                       std::pair<long, decltype(std::ignore)> expected) {
  if (shape.first != expected.first) {
    detail::WrongShape(func, arg, shape, expected);
  }
}

inline void CheckShape(const char *func, const char *arg,
                       std::pair<long, long> const &shape,
                       std::pair<decltype(std::ignore), long> expected) {
  if (shape.second != expected.second) {
    detail::WrongShape(func, arg, shape, expected);
  }
}

inline void CheckShape(const char *func, const char *arg,
                       std::pair<long, long> const &shape,
                       std::pair<long, long> expected) {
  if (shape != expected) {
    detail::WrongShape(func, arg, shape, expected);
  }
}

}  // namespace netket

#endif  // NETKET_EXCEPTIONS_HPP
