#ifndef NETKET_STOPWATCH_HPP
#define NETKET_STOPWATCH_HPP

#include <chrono>

namespace netket {

class Stopwatch
{
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;

public:
    Stopwatch()
        : start_(Clock::now())
    {
    }

    void restart()
    {
        start_ = Clock::now();
    }

    template<typename Duration = std::chrono::microseconds>
    Duration elapsed()
    {
        return std::chrono::duration_cast<Duration>(Clock::now() - start_);
    }
};

}

#endif // NETKET_STOPWATCH_HPP
