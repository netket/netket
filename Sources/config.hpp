#ifndef NETKET_CONFIG_HPP
#define NETKET_CONFIG_HPP

#if defined(WIN32) || defined(_WIN32)
#define NETKET_EXPORT __declspec(dllexport)
#define NETKET_NOINLINE __declspec(noinline)
#define NETKET_FORCEINLINE __forceinline inline
#else
#define NETKET_EXPORT __attribute__((visibility("default")))
#define NETKET_NOINLINE __attribute__((noinline))
#define NETKET_FORCEINLINE __attribute__((always_inline)) inline
#endif

#endif  // NETKET_CONFIG_HPP
