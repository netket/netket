#ifndef NETKET_MATH_HELPERS_HPP
#define NETKET_MATH_HELPERS_HPP

#include <functional>

namespace netket {

/**
 * Returns v if it is in the interval [lo, hi] or the closest of the bounds,
 * if v is outside. Uses comp to compare the values.
 *
 * This is a replacement of std::clamp which was introduced in C++17.
 */
template<class T, class Comparator>
constexpr const T& bound(const T& v, const T& lo, const T& hi, Comparator comp)
{
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

/**
 * Returns v if it is in the interval [lo, hi] or the closest of the bounds,
 * if v is outside. Uses operator< to compare the values.
 *
 * This is a replacement of std::clamp which was introduced in C++17.
 */
template<class T>
constexpr const T& bound(const T& v, const T& lo, const T& hi)
{
    return bound(v, lo, hi, std::less<T>());
}

}

#endif // NETKET_MATH_HELPERS_HPP
