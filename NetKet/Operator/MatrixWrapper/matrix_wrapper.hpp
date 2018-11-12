#ifndef MATRIX_WRAPPER_HPP
#define MATRIX_WRAPPER_HPP

#include <memory>

#include "Operator/abstract_operator.hpp"
#include "Utils/json_helper.hpp"

#include "abstract_matrix_wrapper.hpp"
#include "dense_matrix_wrapper.hpp"
#include "direct_matrix_wrapper.hpp"
#include "sparse_matrix_wrapper.hpp"

namespace netket {

template <class Wrapped>
std::unique_ptr<AbstractMatrixWrapper<Wrapped>> CreateMatrixWrapper(
    const Wrapped& wrapped, const std::string& name = "Sparse") {
  using Ptr = std::unique_ptr<AbstractMatrixWrapper<Wrapped>>;
  if (name == "Dense") {
    return Ptr(new DenseMatrixWrapper<Wrapped>(wrapped));
  } else if (name == "Direct") {
    return Ptr(new DirectMatrixWrapper<Wrapped>(wrapped));
  } else if (name == "Sparse") {
    return Ptr(new SparseMatrixWrapper<Wrapped>(wrapped));
  } else {
    std::stringstream str;
    str << "Unknown matrix wrapper: " << name;
    throw InvalidInputError(str.str());
  }
}

}  // namespace netket

#endif  // MATRIX_WRAPPER_HPP
