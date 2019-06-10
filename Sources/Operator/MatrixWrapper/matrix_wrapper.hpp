#ifndef MATRIX_WRAPPER_HPP
#define MATRIX_WRAPPER_HPP

#include <memory>

#include "Operator/abstract_operator.hpp"

#include "abstract_matrix_wrapper.hpp"
#include "dense_matrix_wrapper.hpp"
#include "direct_matrix_wrapper.hpp"
#include "sparse_matrix_wrapper.hpp"

namespace netket {

template <class State = Eigen::VectorXcd>
std::unique_ptr<AbstractMatrixWrapper<State>> CreateMatrixWrapper(
    const AbstractOperator& wrapped, const std::string& type = "sparse") {
  using Ptr = std::unique_ptr<AbstractMatrixWrapper<State>>;
  if (type == "dense") {
    return Ptr(new DenseMatrixWrapper<State>(wrapped));
  } else if (type == "direct") {
    return Ptr(new DirectMatrixWrapper<State>(wrapped));
  } else if (type == "sparse") {
    return Ptr(new SparseMatrixWrapper<State>(wrapped));
  } else {
    std::stringstream str;
    str << "Unknown matrix wrapper: " << type;
    throw InvalidInputError(str.str());
  }
}

}  // namespace netket

#endif  // MATRIX_WRAPPER_HPP
