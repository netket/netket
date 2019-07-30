// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "Utils/exceptions.hpp"

#include <cassert>
#include <sstream>

namespace netket {
namespace detail {

[[noreturn]] void WrongShape(const char *func, const char *arg, long shape,
                             long expected) {
  assert(shape != expected);
  std::ostringstream msg;
  msg << func << ": " << arg << " has wrong dimension: [" << shape
      << "]; expected [" << expected << "]";
  throw InvalidInputError{msg.str()};
}

[[noreturn]] void WrongShape(const char *func, const char *arg,
                             std::pair<long, long> const &shape,
                             std::pair<long, decltype(std::ignore)> expected) {
  assert(shape.first != expected.first);
  std::ostringstream msg;
  msg << func << ": " << arg << " has wrong dimension: [" << shape.first << ", "
      << shape.second << "]; expected [" << expected.first << ", ?]";
  throw InvalidInputError{msg.str()};
}

[[noreturn]] void WrongShape(const char *func, const char *arg,
                             std::pair<long, long> const &shape,
                             std::pair<decltype(std::ignore), long> expected) {
  assert(shape.second != expected.second);
  std::ostringstream msg;
  msg << func << ": " << arg << " has wrong dimension: [" << shape.first << ", "
      << shape.second << "]; expected [?, " << expected.second << "]";
  throw InvalidInputError{msg.str()};
}

[[noreturn]] void WrongShape(const char *func, const char *arg,
                             std::pair<long, long> const &shape,
                             std::pair<long, long> expected) {
  assert(shape != expected);
  std::ostringstream msg;
  msg << func << ": " << arg << " has wrong dimension: [" << shape.first << ", "
      << shape.second << "]; expected [" << expected.first << ", "
      << expected.second << "]";
  throw InvalidInputError{msg.str()};
}

}  // namespace detail
}  // namespace netket
