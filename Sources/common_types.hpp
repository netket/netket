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
using MatrixXd = Eigen::MatrixXd;
using MatrixXcd = Eigen::MatrixXcd;

}  // namespace netket

#endif  // NETKET_COMMON_TYPES_HPP
