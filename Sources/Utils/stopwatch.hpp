#ifndef NETKET_STOPWATCH_HPP
#define NETKET_STOPWATCH_HPP

#include <chrono>

namespace netket {

class Stopwatch {
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_;

 public:
  Stopwatch() : start_(Clock::now()) {}

  void restart() { start_ = Clock::now(); }

  template <typename Duration = std::chrono::microseconds>
  Duration elapsed() {
    return std::chrono::duration_cast<Duration>(Clock::now() - start_);
  }

  template <typename Duration = std::chrono::microseconds>
  void print_elapsed(std::ostream& out = std::cout) {
    auto elapsed = std::chrono::duration_cast<Duration>(Clock::now() - start_);
    out << elapsed.count() << std::endl;
  }

  template <typename Duration = std::chrono::microseconds>
  void print_elapsed(const std::string& prefix, std::ostream& out = std::cout) {
    out << prefix;
    print_elapsed<Duration>(out);
  }
};

}  // namespace netket

#endif  // NETKET_STOPWATCH_HPP
