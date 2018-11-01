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

#ifndef EXTERNAL_IETL_IETL_VECTORSPACE_H_
#define EXTERNAL_IETL_IETL_VECTORSPACE_H_

#include <ietl/complex.h>
#include <ietl/traits.h>
#include <Eigen/SparseCore>

namespace ietl {

template <class TCoeffs>
class vectorspace {
 public:
  using vector_type = Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>;
  using real_type = typename real_type<TCoeffs>::type;
  using size_type = size_t;
  using scalar_type = TCoeffs;

  explicit vectorspace(size_type dim) : dim_(dim) {}

  inline size_type vec_dimension() const { return dim_; }

  vector_type new_vector() const {
    vector_type v = vector_type::Zero(dim_);
    return v;
  }

  void project(vector_type&) const {}

 private:
  size_type dim_;
};

template <class VS>
void project(typename vectorspace_traits<VS>::vector_type& v, const VS& vs) {
  vs.project(v);
}

template <class VS>
typename ietl::vectorspace_traits<VS>::vector_type new_vector(const VS& vs) {
  return vs.new_vector();
}

template <class VS>
typename ietl::vectorspace_traits<VS>::size_type vec_dimension(const VS& vs) {
  return vs.vec_dimension();
}
}  // namespace ietl

#endif  // EXTERNAL_IETL_IETL_VECTORSPACE_H_
