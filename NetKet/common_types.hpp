#ifndef NETKET_COMMON_TYPES_HPP
#define NETKET_COMMON_TYPES_HPP
/**
 * This header contains standard type aliases to be used throughout the NetKet
 * codebase.
 */

#include <complex>
#include <cstddef>

namespace netket {

using Index = std::ptrdiff_t;
using Complex = std::complex<double>;

}  // namespace netket

#endif  // NETKET_COMMON_TYPES_HPP
