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

/* author: Alexander Wietek, 2018/10/22 */

#ifndef EXTERNAL_IETL_IETL_RANDOMGENERATOR_H_
#define EXTERNAL_IETL_IETL_RANDOMGENERATOR_H_

#include <random>

namespace ietl {

  template <class random_engine_t, class distribution_t>
  class random_generator {
  public:
    random_generator(distribution_t distribution, int seed)
      : distribution_(distribution)
    { engine_.seed(seed); }

    typename distribution_t::result_type operator() ()
    { return distribution_(engine_); }

  private:
    random_engine_t engine_;
    distribution_t distribution_;
  };
}

#endif  // EXTERNAL_IETL_IETL_RANDOMGENERATOR_H_
