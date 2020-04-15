#ifndef NETKET_COMMON_TYPES_HPP
#define NETKET_COMMON_TYPES_HPP
/**
 * This header contains standard type aliases to be used throughout the NetKet
 * codebase.
 */

#include <complex>
#include <cstddef>

#include <Eigen/Core>

namespace netket {

using Index = std::ptrdiff_t;
using Complex = std::complex<double>;

using VectorXd = Eigen::VectorXd;
using VectorXcd = Eigen::VectorXcd;

template <class T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using MatrixXd = Matrix<double>;
using MatrixXcd = Matrix<Complex>;

template <class T>
using RowMatrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using RowMatrixXd = RowMatrix<double>;
using RowMatrixXcd = RowMatrix<Complex>;

}  // namespace netket

#endif  // NETKET_COMMON_TYPES_HPP
