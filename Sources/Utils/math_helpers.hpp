#ifndef NETKET_MATH_HELPERS_HPP
#define NETKET_MATH_HELPERS_HPP

#include <cmath>
#include <functional>
#include <limits>

namespace netket {

/**
 * Returns v if it is in the interval [lo, hi] or the closest of the bounds,
 * if v is outside. Uses comp to compare the values.
 *
 * This is a replacement of std::clamp which was introduced in C++17.
 */
template <class T, class Comparator>
constexpr const T& bound(const T& v, const T& lo, const T& hi,
                         Comparator comp) {
  return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

/**
 * Returns v if it is in the interval [lo, hi] or the closest of the bounds,
 * if v is outside. Uses operator< to compare the values.
 *
 * This is a replacement of std::clamp which was introduced in C++17.
 */
template <class T>
constexpr const T& bound(const T& v, const T& lo, const T& hi) {
  return bound(v, lo, hi, std::less<T>());
}

inline bool RelativelyEqual(double a, double b, double maxRelativeDiff) {
  const double difference = std::abs(a - b);
  // Scale to the largest value.
  a = std::abs(a);
  b = std::abs(b);
  const double scaledEpsilon = maxRelativeDiff * std::max(a, b);
  return difference <= scaledEpsilon;
}

inline bool CheckProductOverflow(int a, int b) {
  if (a == 0 || b == 0)
    return false;
  else
    return std::log(std::abs(a)) + std::log(std::abs(b)) >
           std::log(std::numeric_limits<int>::max());
}

inline bool CheckSumOverflow(int a, int b) {
  if (b < 0) {
    if (a >= 0)
      return false;
    else {
      return abs(a) > std::numeric_limits<int>::max() - abs(b);
    }
  } else
    return a > std::numeric_limits<int>::max() - b;
}

}  // namespace netket

#endif  // NETKET_MATH_HELPERS_HPP
